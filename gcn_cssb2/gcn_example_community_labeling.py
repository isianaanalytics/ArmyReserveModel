# generates community labeling figures for CSSB example.
# figure1.png is the initial graph; graph clusters in output_anim.mp4
# figure2.png shows output lost for training and test sets
# credit https://bagheri365.github.io/blog/Graph-Convolutional-Networks-(GCNs)-from-Scratch/
# -Virgilio Villacorta, 17 March 2024


import numpy as np
from scipy.linalg import sqrtm
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import matplotlib.pyplot as plt
from matplotlib import animation
#get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML

class GradDescentOptim():
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd
        self._y_pred = None
        self._y_true = None
        self._out = None
        self.bs = None
        self.train_nodes = None

    def __call__(self, y_pred, y_true, train_nodes=None):
        self.y_pred = y_pred
        self.y_true = y_true

        if train_nodes is None:
            self.train_nodes = np.arange(y_pred.shape[0])
        else:
            self.train_nodes = train_nodes

        self.bs = self.train_nodes.shape[0]

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, y):
        self._out = y


class GCNLayer():
    def __init__(self, n_inputs, n_outputs, activation=None, name=''):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.activation = activation
        self.name = name

    def __repr__(self):
        return f"GCN: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"

    def forward(self, A, X, W=None):
        """
        Assumes A is (bs, bs) adjacency matrix and X is (bs, D),
            where bs = "batch size" and D = input feature length
        """
        self._A = A
        self._X = (A @ X).T # for calculating gradients.  (D, bs)

        if W is None:
            W = self.W

        H = W @ self._X # (h, D)*(D, bs) -> (h, bs)
        if self.activation is not None:
            H = self.activation(H)
        self._H = H # (h, bs)
        return self._H.T # (bs, h)

    def backward(self, optim, update=True):
        dtanh = 1 - np.asarray(self._H.T)**2 # (bs, out_dim)
        d2 = np.multiply(optim.out, dtanh)  # (bs, out_dim) *element_wise* (bs, out_dim)

        self.grad = self._A @ d2 @ self.W # (bs, bs)*(bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
        optim.out = self.grad

        dW = np.asarray(d2.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, D) -> (out_dim, D)
        dW_wd = self.W * optim.wd / optim.bs # weight decay update

        if update:
            self.W -= (dW + dW_wd) * optim.lr

        return dW + dW_wd


class SoftmaxLayer():
    def __init__(self, n_inputs, n_outputs, name=''):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.b = np.zeros((self.n_outputs, 1))
        self.name = name
        self._X = None # Used to calculate gradients

    def __repr__(self):
        return f"Softmax: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"

    def shift(self, proj):
        shiftx = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, X, W=None, b=None):
        """Compute the softmax of vector x in a numerically stable way.

        X is assumed to be (bs, h)
        """
        self._X = X.T
        if W is None:
            W = self.W
        if b is None:
            b = self.b

        proj = np.asarray(W @ self._X) + b # (out, h)*(h, bs) = (out, bs)
        return self.shift(proj).T # (bs, out)

    def backward(self, optim, update=True):
        # should take in optimizer, update its own parameters and update the optimizer's "out"
        # Build mask on loss
        train_mask = np.zeros(optim.y_pred.shape[0])
        train_mask[optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1))

        # derivative of loss w.r.t. activation (pre-softmax)
        d1 = np.asarray((optim.y_pred - optim.y_true)) # (bs, out_dim)
        d1 = np.multiply(d1, train_mask) # (bs, out_dim) with loss of non-train nodes set to zero

        self.grad = d1 @ self.W # (bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
        optim.out = self.grad

        dW = (d1.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, in_dim) -> (out_dim, in_dim)
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs # (out_dim, 1)

        dW_wd = self.W * optim.wd / optim.bs # weight decay update

        if update:
            self.W -= (dW + dW_wd) * optim.lr
            self.b -= db.reshape(self.b.shape) * optim.lr

        return dW + dW_wd, db.reshape(self.b.shape)

# # The Model
class GCNNetwork():
    def __init__(self, n_inputs, n_outputs, n_layers, hidden_sizes, activation, seed=0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        np.random.seed(seed)

        self.layers = list()
        # Input layer
        gcn_in = GCNLayer(n_inputs, hidden_sizes[0], activation, name='in')
        self.layers.append(gcn_in)

        # Hidden layers
        for layer in range(n_layers):
            gcn = GCNLayer(self.layers[-1].W.shape[0], hidden_sizes[layer], activation, name=f'h{layer}')
            self.layers.append(gcn)

        # Output layer
        sm_out = SoftmaxLayer(hidden_sizes[-1], n_outputs, name='sm')
        self.layers.append(sm_out)

    def __repr__(self):
        return '\n'.join([str(l) for l in self.layers])

    def embedding(self, A, X):
        # Loop through all GCN layers
        H = X
        for layer in self.layers[:-1]:
            H = layer.forward(A, H)
        return np.asarray(H)

    def forward(self, A, X):
        # GCN layers
        H = self.embedding(A, X)

        # Softmax
        p = self.layers[-1].forward(H)

        return np.asarray(p)


def draw_kkl(nx_G, label_map, node_color, pos=None, **kwargs):
    fig, ax = plt.subplots(figsize=(10,10))
    if pos is None:
        pos = nx.spring_layout(nx_G, k=5/np.sqrt(nx_G.number_of_nodes()))

    nx.draw(
        nx_G, pos, with_labels=label_map is not None,
        labels=label_map,
        node_color=node_color,
        ax=ax, **kwargs)


def glorot_init(nin, nout):
    sd = np.sqrt(6.0 / (nin + nout))
    return np.random.uniform(-sd, sd, size=(nin, nout))


def xent(pred, labels):
    return -np.log(pred)[np.arange(pred.shape[0]), np.argmax(labels, axis=1)]


def norm_diff(dW, dW_approx):
    return np.linalg.norm(dW - dW_approx) / (np.linalg.norm(dW) + np.linalg.norm(dW_approx))


# ## Gradient checking on Softmax layer
def get_grads(inputs, layer, argname, labels, eps=1e-4, wd=0):
    cp = getattr(layer, argname).copy()
    cp_flat = np.asarray(cp).flatten()
    grads = np.zeros_like(cp_flat)
    n_parms = cp_flat.shape[0]
    for i, theta in enumerate(cp_flat):
        #print(f"Parm {argname}_{i}")
        theta_cp = theta

        # J(theta + eps)
        cp_flat[i] = theta + eps
        cp_tmp = cp_flat.reshape(cp.shape)
        predp = layer.forward(*inputs, **{argname: cp_tmp})
        wd_term = wd/2*(cp_flat**2).sum() / labels.shape[0]
        #print(wd_term)
        Jp = xent(predp, labels).mean() + wd_term

        # J(theta - eps)
        cp_flat[i] = theta - eps
        cp_tmp = cp_flat.reshape(cp.shape)
        predm = layer.forward(*inputs, **{argname: cp_tmp})
        wd_term = wd/2*(cp_flat**2).sum() / labels.shape[0]
        #print(wd_term)
        Jm = xent(predm, labels).mean() + wd_term

        # grad
        grads[i] = ((Jp - Jm) / (2*eps))

        # Back to normal
        cp_flat[i] = theta

    return grads.reshape(cp.shape)


# ## Gradient checking on GCN layer
def get_gcn_grads(inputs, gcn, sm_layer, labels, eps=1e-4, wd=0):
    cp = gcn.W.copy()
    cp_flat = np.asarray(cp).flatten()
    grads = np.zeros_like(cp_flat)
    n_parms = cp_flat.shape[0]
    for i, theta in enumerate(cp_flat):
        theta_cp = theta

        # J(theta + eps)
        cp_flat[i] = theta + eps
        cp_tmp = cp_flat.reshape(cp.shape)
        pred = sm_layer.forward(gcn.forward(*inputs, W=cp_tmp))
        w2 = (cp_flat**2).sum()+(sm_layer.W.flatten()**2).sum()
        Jp = xent(pred, labels).mean() + wd/(2*labels.shape[0])*w2

        # J(theta - eps)
        cp_flat[i] = theta - eps
        cp_tmp = cp_flat.reshape(cp.shape)
        pred = sm_layer.forward(gcn.forward(*inputs, W=cp_tmp))
        w2 = (cp_flat**2).sum()+(sm_layer.W.flatten()**2).sum()
        Jm = xent(pred, labels).mean() + wd/(2*labels.shape[0])*w2

        # grad
        grads[i] = ((Jp - Jm) / (2*eps))

        # Back to normal
        cp_flat[i] = theta

    return grads.reshape(cp.shape)

# ## Gradient checking on inputs
def get_gcn_input_grads(A_hat, X, gcn, sm_layer, labels, eps=1e-4):
    cp = X.copy()
    cp_flat = np.asarray(cp).flatten()
    grads = np.zeros_like(cp_flat)
    n_parms = cp_flat.shape[0]
    for i, x in enumerate(cp_flat):
        x_cp = x

        # J(theta + eps)
        cp_flat[i] = x + eps
        cp_tmp = cp_flat.reshape(cp.shape)
        pred = sm_layer.forward(gcn.forward(A_hat, cp_tmp))
        Jp = xent(pred, labels).mean()

        # J(theta - eps)
        cp_flat[i] = x - eps
        cp_tmp = cp_flat.reshape(cp.shape)
        pred = sm_layer.forward(gcn.forward(A_hat, cp_tmp))
        Jm = xent(pred, labels).mean()

        # grad
        grads[i] = ((Jp - Jm) / (2*eps))

        # Back to normal
        cp_flat[i] = x

    return grads.reshape(cp.shape)

G = nx.Graph()
mapping = {
    0:'XO',
    1:'SPO',
    2:'S3',
    3:'S4',
    4:'Tm Chief 1',
    5:'Tm Chief 2',
    6:'Tm Chief 3',
    7:'Tm Chief 4',
    8:'HHD CO',
    9: 'HHD XO',
    10:'SMC CO',
    11:'MTC CO',
    12:'SMC 1PL',
    13:'SMC 2PL',
    14:'SMC 3PL',
    15:'MTC 1PL',
    16:'MTC 2PL',
    17:'MTC 3PL',
    18:'S3 NCOIC',
    19:'S4 NCOIC',
    20:'SPO NCOIC',
    21:'Tm1 NCOIC',
    22:'Tm2 NCOIC',
    23:'Tm3 NCOIC',
    24:'Tm4 NCOIC',
    25:'SMC 1PSG',
    26:'SMC 2PSG',
    27:'SMC 3PSG',
    28:'MTC 1PSG',
    29:'MTC 2PSG',
    30:'MTC 2PSG',
    31:'HHD Det SGT',
    32:'SMC 1SG',
    33:'MTC 1SG'
}

#adding edges
G.add_edges_from([
  (0,1), (1,0), (2,0),

  (2,3), (3,2),

  (3,4), (3,5), (3,6), (3,7), (3,8), (3,10), (3,11),

  (4,1), (4,8), (4,10), (4,11), (5,1), (6,1), (7,1),

  (4,2), (4,3),(4,5), (4,6), (4,7), (5,3),

  (5,4), (5,6), (5,7), (5,8), (5,10), (5,11), (6,3),

  (6,4), (6,5), (6,7), (6,8), (6,10), (6,11), (7,3), (7,4),

  (7,5), (7,6), (7,8), (7,10),(7,11), (8,3), (8,4), (8,5),

  (8,6), (8,7), (8,10), (8,2), (9,8), (10,3), (10,8), (8,11),

  (11,3), (11,8), (12,9), (12,10), (12,13),

  (12,14),(12,15), (12,16),(12,17), (13,9), (13,10),

  (13,12), (13,14),(13,15), (13,16),(13,17), (14,9), (14,10),

  (14,12), (14,13), (14,15), (14,16),(14,17), (15,9),

  (15,11), (15,12), (15,13), (15,14), (15,16),(15,17), (16,9),

  (16,11), (16,12), (16,13), (16,14),(16,15), (16,17), (17,9),

  (17,11), (17,12), (17,13), (17,14),(17,15), (17,16),

  (18,2), (18,20), (18,21), (18,22), (18,23),(18,24),(18,25),(18,26),

  (18,27), (18,31), (19,3), (19,20), (19,20), (19,21), (19,22),(19,23),

  (19,24), (19,28), (19,29), (19,30), (19,31),

  (20,1), (20,31),(20,32),

  (20,33), (21,18), (21,19), (21,20), (21,22), (21,23), (21,24), (21,25), (21,26),

  (21,27), (21,31), (22,20), (22,21), (22,23), (22,24), (22,25), (22,26),

  (22,27), (22,28), (22,29), (22,30), (22,31), (23,20), (23,21), (23,22), (23,24),

  (23,28), (23,29), (23,30), (23,31), (24,20), (24,21), (24,22), (24,23), (24,25),

  (24,26), (24,27), (24,31), (25,12), (25,21), (25,22), (25,23), (25,24),

  (25,26), (25,27), (25,31), (25,32), (26,13), (26,21), (26,22), (26,23),

  (26,24), (26,25), (26,27), (26,31), (27,14), (27,21), (27,22), (27,23),

  (27,24), (27,25), (27,26), (27,28), (27,29), (27,30), (27,31), (28,15),

  (28,21), (28,22), (28,23), (28,24), (28,29), (28,30), (28,31),

  (28,32), (29,16), (29,21), (29,22), (29,23), (29,24),

  (29,28), (29,30), (29,31), (30,17), (30,21), (30,22), (30,23),

  (30,24), (30,28), (30,29), (30,31), (31,8), (31,9), (31,20),

  (31,33), (32,10), (32,20), (32,31), (32,33), (33,11), (33,20),

  (33,31), (33,32)])

g = G

g.number_of_nodes(), g.number_of_edges()

communities = greedy_modularity_communities(g, resolution = 1.3)


colors = np.zeros(g.number_of_nodes())
for i, com in enumerate(communities):
    colors[list(com)] = i

n_classes = np.unique(colors).shape[0]

labels = np.eye(n_classes)[colors.astype(int)]


club_labels = nx.get_node_attributes(g,'club')


_ = draw_kkl(g, None, colors, cmap='gist_rainbow', edge_color='gray')


fig, ax = plt.subplots(figsize=(10,10))
#pos = nx.spring_layout(g, k=5/np.sqrt(g.number_of_nodes()))
pos = nx.circular_layout(g)
#kwargs = {"cmap": 'gist_rainbow', "edge_color":'gray'}
nx.draw(
    g, pos, with_labels=True,
    node_color='cyan',
    ax=ax) #, **kwargs)
#plt.savefig('karate_club_graph.png', bbox_inches='tight', transparent=True)
fig.savefig('figure1.png')
# In[11]:


A = nx.adjacency_matrix(g) #nx.to_numpy_array(g) #nx.to_numpy_matrix(g)


A_mod = A + np.eye(g.number_of_nodes()) # add self-connections

D_mod = np.zeros_like(A_mod)
np.fill_diagonal(D_mod, np.asarray(A_mod.sum(axis=1)).flatten())


D_mod_invroot = np.linalg.inv(sqrtm(D_mod))

A_hat = D_mod_invroot @ A_mod @ D_mod_invroot


X = np.eye(g.number_of_nodes())



gcn1 = GCNLayer(g.number_of_nodes(), 2, activation=np.tanh, name='1')
sm1 = SoftmaxLayer(2, n_classes, "SM")
opt = GradDescentOptim(lr=0, wd=1.)


gcn1_out = gcn1.forward(A_hat, X)
opt(sm1.forward(gcn1_out), labels)



dW_approx = get_grads((gcn1_out,), sm1, "W", labels, eps=1e-4, wd=opt.wd)
db_approx = get_grads((gcn1_out,), sm1, "b", labels, eps=1e-4, wd=opt.wd)


# Get gradients on Linear Softmax layer
dW, db = sm1.backward(opt, update=False)


assert norm_diff(dW, dW_approx) < 1e-7
assert norm_diff(db, db_approx) < 1e-7


dW2 = gcn1.backward(opt, update=False)

dW2_approx = get_gcn_grads((A_hat, X), gcn1, sm1, labels, eps=1e-4, wd=opt.wd)


assert norm_diff(dW2, dW2_approx) < 1e-7


dX_approx = get_gcn_input_grads(A_hat, X, gcn1, sm1, labels, eps=1e-4)


assert norm_diff(gcn1.grad/A_hat.shape[0], dX_approx) < 1e-7


gcn_model = GCNNetwork(
    n_inputs=g.number_of_nodes(),
    n_outputs=n_classes,
    n_layers=2,
    hidden_sizes=[22, 2],
    #hidden_sizes=[10, 2],
    activation=np.tanh,
    seed=100,
)
gcn_model


y_pred = gcn_model.forward(A_hat, X)
embed = gcn_model.embedding(A_hat, X)
xent(y_pred, labels).mean()


pos = {i: embed[i,:] for i in range(embed.shape[0])}
_ = draw_kkl(g, None, colors, pos=pos, cmap='gist_rainbow', edge_color='gray')


# ### Training

range(labels.shape[0])


train_nodes = np.array([4, 15, 23, 31]) #([3, 5, 10,11, 15, 20, 28])
test_nodes = np.array([i for i in range(labels.shape[0])]) # if i not in train_nodes])
opt2 = GradDescentOptim(lr=2e-2, wd=2.5e-2)


embeds = list()
accs = list()
train_losses = list()
test_losses = list()

loss_min = 1e6
es_iters = 0
es_steps = 50
# lr_rate_ramp = 0 #-0.05
# lr_ramp_steps = 1000

for epoch in range(15000):

    y_pred = gcn_model.forward(A_hat, X)

    opt2(y_pred, labels, train_nodes)

#     if ((epoch+1) % lr_ramp_steps) == 0:
#         opt2.lr *= 1+lr_rate_ramp
#         print(f"LR set to {opt2.lr:.4f}")

    for layer in reversed(gcn_model.layers):
        layer.backward(opt2, update=True)

    embeds.append(gcn_model.embedding(A_hat, X))
    # Accuracy for non-training nodes
    acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
        [i for i in range(labels.shape[0]) if i not in train_nodes]
    ]
    accs.append(acc.mean())

    loss = xent(y_pred, labels)
    loss_train = loss[train_nodes].mean()
    loss_test = loss[test_nodes].mean()

    train_losses.append(loss_train)
    test_losses.append(loss_test)

    if loss_test < loss_min:
        loss_min = loss_test
        es_iters = 0
    else:
        es_iters += 1

    if es_iters > (15*es_steps):
        print("Early stopping!")
        break

    if epoch % 100 == 0:
        print(f"Epoch: {epoch+1}, Train Loss: {loss_train:.3f}, Test Loss: {loss_test:.3f}")

train_losses = np.array(train_losses)
test_losses = np.array(test_losses)


fig, ax = plt.subplots()
ax.plot(np.log10(train_losses), label='Train')
ax.plot(np.log10(test_losses), label='Test')
ax.legend()
ax.grid()

fig.savefig('figure2.png')

accs[-1]


pos = {i: embeds[-1][i,:] for i in range(embeds[-1].shape[0])}
_ = draw_kkl(g, None, colors, pos=pos, cmap='gist_rainbow', edge_color='gray')


N = 1500
snapshots = np.linspace(0, len(embeds)-1, N).astype(int)


# Build plot
fig, ax = plt.subplots(figsize=(10, 10))
kwargs = {'cmap': 'gist_rainbow', 'edge_color': 'gray', }#'node_size': 55}

def update(idx):
    ax.clear()
    embed = embeds[snapshots[idx]]
    pos = {i: embed[i,:] for i in range(embed.shape[0])}
    nx.draw(g, pos, node_color=colors, with_labels = True, ax=ax, **kwargs)
anim = animation.FuncAnimation(fig, update, frames=snapshots.shape[0], interval=10, repeat=False)


HTML(anim.to_html5_video())


anim.save('output_anim.mp4', dpi=300)
