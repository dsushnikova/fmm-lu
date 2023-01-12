import numpy as np
from numba import jit
# from h2tools.cluster_tree import SmartIndex
from copy import deepcopy as dc

class SmartIndex(object):
    """
    Stores only view to index and information about each node.

    It is only used in `ClusterTree` class for convenient work with
    indexes. Main reason this is implemented separately from
    `ClusterTree` is easily readable syntax: `index[key]` returns view
    to subarray of array `index`, corresponding to indices of node
    `key`.

    Parameters
    ----------
    size : integer
        Number of objects in cluster

    Attributes
    ----------
    index: 1-dimensional array
        Permutation array such, that indexes of objects, corresponding
        to the same subcluster, are located one after each other.
    node: list of tuples
        Indexes of `i`-th node of cluster tree are
        `index[node[i][0]:node[i][1]]`.
    """

    def __init__(self, size):
        self.index = np.arange(size, dtype=np.uint64)
        self.node = [(0, size)]

    def __getitem__(self, key):
        """Get indices for cluster `key`."""
        return self.index[slice(*self.node[key])]

    def __setitem__(self, key, value):
        """
        Set indices for cluster `key`.

        Changes only main index array.
        """
        self.index[slice(*self.node[key])] = value

    def add_node(self, parent, node):
        """Add node, that corresponds to `index[node[0]:node[1]]`."""
        start = self.node[parent][0]+node[0]
        stop = self.node[parent][0]+node[1]
        self.node.append((start, stop))

    def __len__(self):
        return len(self.node)
class Data(object):
    def __init__(self, ndim, count, vertex, close_r='1box'):
        self.ndim = ndim
        self.count = count
        self.vertex = vertex
        self.close_r = close_r
    def check_far(self, self_aux, other_aux):
        return Data.fast_check_far_ndim(self_aux, other_aux, self.ndim, self.close_r)
    def fast_check_far_ndim(self_aux, other_aux, ndim, close_r):
        if close_r == '1box':
            if ndim == 2:
                corners_self = [np.array([self_aux[0,0], self_aux[0,1]]),
                                np.array([self_aux[1,0], self_aux[0,1]]),
                                np.array([self_aux[0,0], self_aux[1,1]]),
                                np.array([self_aux[1,0], self_aux[1,1]])]
                corners_other = [np.array([other_aux[0,0], other_aux[0,1]]),
                                 np.array([other_aux[1,0], other_aux[0,1]]),
                                 np.array([other_aux[0,0], other_aux[1,1]]),
                                 np.array([other_aux[1,0], other_aux[1,1]])]
                for i in corners_self:
                    for j in corners_other:
                        if np.array_equal(i,j):
                            return False

                return True
            elif ndim == 3:
                corners_self = [np.array([self_aux[0,0], self_aux[0,1], self_aux[0,2]]),
                                np.array([self_aux[0,0], self_aux[0,1], self_aux[1,2]]),
                                np.array([self_aux[0,0], self_aux[1,1], self_aux[0,2]]),
                                np.array([self_aux[0,0], self_aux[1,1], self_aux[1,2]]),
                                np.array([self_aux[1,0], self_aux[0,1], self_aux[0,2]]),
                                np.array([self_aux[1,0], self_aux[0,1], self_aux[1,2]]),
                                np.array([self_aux[1,0], self_aux[1,1], self_aux[0,2]]),
                                np.array([self_aux[1,0], self_aux[1,1], self_aux[1,2]])]
                corners_other = [np.array([other_aux[0,0], other_aux[0,1], other_aux[0,2]]),
                                 np.array([other_aux[0,0], other_aux[0,1], other_aux[1,2]]),
                                 np.array([other_aux[0,0], other_aux[1,1], other_aux[0,2]]),
                                 np.array([other_aux[0,0], other_aux[1,1], other_aux[1,2]]),
                                 np.array([other_aux[1,0], other_aux[0,1], other_aux[0,2]]),
                                 np.array([other_aux[1,0], other_aux[0,1], other_aux[1,2]]),
                                 np.array([other_aux[1,0], other_aux[1,1], other_aux[0,2]]),
                                 np.array([other_aux[1,0], other_aux[1,1], other_aux[1,2]])]
                for i in corners_self:
                    for j in corners_other:
                        if np.allclose(i,j):
                            return False

                return True
        elif type(close_r) == float:
            diam0 = 0.
            diam1 = 0.
            dist = 0.
            for i in range(ndim):
                tmp = self_aux[0, i]-self_aux[1, i]
                diam0 += tmp*tmp
                tmp = other_aux[0, i]-other_aux[1, i]
                diam1 += tmp*tmp
                tmp = self_aux[0, i]+self_aux[1, i]-other_aux[0, i]-other_aux[1, i]
                dist += tmp*tmp
            dist *= 0.25
            return dist > diam0 * close_r and dist > diam1*close_r
        else:
            raise NameError('Wrong close_r')
    def compute_aux(self, index):
        tmp_particles = self.vertex[:,index]

        xyzmin = np.min(tmp_particles,axis=1)
        xyzmax = np.max(tmp_particles,axis=1)
        bs = xyzmax-xyzmin
        bs = np.max(bs)*1.01 # Fudge factor to account for some rounding issue
        xyzmaxuse = xyzmin + bs
        return np.array([xyzmin,xyzmaxuse])
    def divide(self, index):
        vertex = self.vertex[:, index]
        center = vertex.mean(axis=1)
        vertex -= center.reshape(-1, 1)
        normal = np.linalg.svd(vertex, full_matrices=0)[0][:,0]
        scal_dot = normal.dot(vertex)
        scal_sorted = scal_dot.argsort()
        scal_dot = scal_dot[scal_sorted]
        k = scal_dot.searchsorted(0)
        return scal_sorted, [0, k, scal_sorted.size]
    def half_box(self, index, ax, mid_point):
        ndim = self.ndim
        vertex = self.vertex[:, index]
        center = mid_point#vertex.mean(axis=1)
        vertex -= center.reshape(-1, 1)
        normal = np.zeros(ndim)
        normal[ax] = 1.
        scal_dot = normal.dot(vertex)
        scal_sorted = scal_dot.argsort()
        scal_dot = scal_dot[scal_sorted]
        k = scal_dot.searchsorted(0)
        return scal_sorted, [0, k, scal_sorted.size]
    def __len__(self):
        return self.count
class Tree(object):
    def __init__(self, data, block_size, point_based_tree = True, num_child_tree = 'hyper'):
        self.block_size = block_size
        self.data = data
        self.index = SmartIndex(len(data))
        self.parent = [-1]
        self.child = [[]]
        self.leaf = [0]
        self.level = [0, 1]
        self.num_levels = 0
        self.num_leaves = 1
        self.num_nodes = 1
        self.point_based_tree = point_based_tree
        self.num_child_tree = num_child_tree
        if num_child_tree == 'hyper':
            self.nchild = 2 ** data.ndim
        elif num_child_tree == 2:
            self.nchild = num_child_tree
        else:
            print(f'Number of children = {num_child_tree} is not suported, # children changed to 2')
            self.nchild = 2
    def divide_space(self, key):
        ndim = self.data.ndim
        index = self.index[key]
        box_list = []
        for i in range(self.nchild):
            box_list.append(dc(self.aux[key]))
        if self.num_child_tree == 'hyper':
            for i in range(ndim):
                mid_point = (self.aux[key][0, i] + self.aux[key][1, i]) / 2
                for ii in range(len(box_list)):
                    if self.check(ii, i, ndim):
                        box_list[ii][1,i] = mid_point
                    else:
                        box_list[ii][0,i] = mid_point
            index_list_old = []
            for i in range(2**ndim):
                self.aux.append(box_list[i])
                index_list_old.append([])
            vertex = self.data.vertex[:, index]
            for i_v in range(vertex.shape[1]):
                v_in_b = 0
                for i_aux in range(2**ndim):
                    vertex_in_box = 1
                    for nd in range(ndim):
                        vertex_in_box = vertex_in_box and (vertex[nd,i_v] >= box_list[i_aux][0,nd]) and (vertex[nd,i_v] <= box_list[i_aux][1,nd])
                    if vertex_in_box and v_in_b == 0:
                        index_list_old[i_aux].append(index[i_v])
                        v_in_b += 1
                if v_in_b != 1:
                    for i_aux in range(2**ndim):
                        print(f'n box: {i_aux}, box: {box_list[i_aux]} \nlast ind: {index_list_old[i_aux]}')
                    raise NameError(f'{i_v, v_in_b}, v:{vertex[:,i_v]}')


            index_res = np.array([])
            list_k = [0]
            for local_index in index_list_old:
                local_index = np.array(local_index)
                list_k.append(list_k[-1]+local_index.shape[0])
                index_res = np.hstack((index_res,local_index))
            return index_res.astype(int), list_k
        else:
            ax = len(self.level)%2
            mid_point = (self.aux[key][0, ax] + self.aux[key][1, ax]) / 2
            box_list[0][1,ax] = mid_point
            box_list[1][0,ax] = mid_point
            for i in range(2):
                self.aux.append(box_list[i])
            l = len(self.level)
            new_index, subclusters = self.data.half_box(index, l%2, mid_point)
            new_index = index[new_index]
            return new_index, subclusters
    def divide_point(self, key):
        ndim = self.data.ndim
        index = self.index[key]
        if self.num_child_tree == 'hyper':
            index_list_old = [self.index[key]]
            list_k = [0]
            ndim = self.data.ndim
            for i in range(ndim):
                index_list_new = []
                for index in index_list_old:
                    new_index, subclusters = self.data.divide(index)
                    new_index = index[new_index]
                    index_list_new.append(new_index[:subclusters[1]])
                    index_list_new.append(new_index[subclusters[1]:])
                index_list_old = dc(index_list_new)
            index_res = np.array([])
            for local_index in index_list_old:
                list_k.append(list_k[-1]+np.array(local_index).shape[0])
                index_res = np.hstack((index_res,local_index))
            new_index = dc(index_res.astype(int))
            subclusters = dc(list_k)
        else:
            index = self.index[key]
            new_index, subclusters = self.data.divide(index)
            new_index = index[new_index]

        last_ind = subclusters[0]
        for i in range(len(subclusters)-1):
            next_ind = subclusters[i+1]
            self.aux.append(self.data.compute_aux(new_index[last_ind:next_ind]))
            last_ind = next_ind
        return new_index, subclusters
    def check(self, n, dim, ndim):
        for _ in range(ndim - dim):
            res = n % 2
            n = n // 2
        return res==0
    def divide(self, key):
        ndim = self.data.ndim
        index = self.index[key]
        # d = 1/0
        if self.point_based_tree:
            new_index, subclusters = self.divide_point(key)
        else:
            new_index, subclusters = self.divide_space(key)
        test_index = new_index.copy()
        test_index.sort()
        last_ind = subclusters[0]
        for i in range(len(subclusters)-1):
            next_ind = subclusters[i+1]
            if next_ind < last_ind:
                raise NameError("children indices must be one after other")
            self.index.add_node(key, (last_ind, next_ind))
            last = len(self.parent)
            self.parent.append(key)
            if self.child[key]:
                self.num_leaves += 1
            self.num_nodes += 1
            self.child[key].append(last)
            self.child.append([])

            last_ind = next_ind
        if next_ind != test_index.size:
            raise Error("Sum of sizes of children must be the same as"
                " size of the parent")
        self.index[key] = new_index
    def is_far(self, i, other_tree, j):
        if i <= j:
            result = self.data.check_far(self.aux[i], other_tree.aux[j])
        else:
            result = other_tree.data.check_far(other_tree.aux[j], self.aux[i])
        return result
    def __len__(self):
        return len(self.parent)
