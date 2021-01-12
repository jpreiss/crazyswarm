from copy import deepcopy
import networkx as nx
import numpy as np
import scipy

from opt_utils.optimization import *
from opt_utils.formation import *


DEFAULT_BBOX = np.array([(-30, 30), (-30, 30), (10, 100)])


class DKFNetwork:
    def __init__(self,
                 nodes,
                 weights,
                 G,
                 targets):
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, weights, 'weights')
        self.targets = targets  # the true targets

        # TRACKERS
        self.failures = {}
        self.adjacencies = {}
        self.weighted_adjacencies = {}
        self.errors = {}
        self.max_trace_cov = {}
        self.mean_trace_cov = {}
        self.surveillance_quality = {}


    """
    Simulation Operations
    """
    def step(self, timestep, input, opt, fail_node,
             failure=False, base=False, L=10, noise_mult=1, known_input=False):
        nodes = nx.get_node_attributes(self.network, 'node')
        i = timestep
        ins = input

        """
        True Target Update With Inputs
        """
        next_input = []
        for t, target in enumerate(self.targets):
            new_input = deepcopy(ins[t])
            next_state = target.get_next_state(input=ins[t])
            new_input = check_oob(next_state, new_input)
            target.next_state(input=new_input)
            next_input.append(new_input)

        """
        Local Target Estimation
        """
        for id, n in nodes.items():
            if known_input:
                n.predict(len(nodes))
            else:
                n.predict(len(nodes), inputs=next_input)
            ms = n.get_measurements(self.targets)
            if failure:
                ms = [m + np.random.random(m.shape) * noise_mult for m in ms]
            n.update(ms)
            n.update_trackers(i)

        """
        Init Consensus
        """
        for id, n in nodes.items():
            n.init_consensus()

        """
        Do Optimization and Formation Synthesis
        """
        if failure and not base:
            if opt == 'agent':
                self.do_agent_opt(fail_node)
            elif opt == 'team':
                self.do_team_opt(fail_node)
            elif opt == 'greedy':
                self.do_greedy_opt(fail_node)

            # Random strategy
            else:
                self.do_random_opt(fail_node)

            # Formation Synthesis
            current_coords = {nid: n.position for nid, n in nodes.items()}
            fov = {nid: n.fov for nid, n in nodes.items()}
            Rs = {nid: n.R for nid, n in nodes.items()}

            new_coords, sq = generate_coords(self.adjacency_matrix(),
                                             current_coords, fov, Rs)
            self.surveillance_quality[i] = sq
            if new_coords:
                for id, n in nodes.items():
                    n.update_position(new_coords[id])

        if failure and base:
            # Just get current surveillance quality
            current_coords = {nid: n.position for nid, n in nodes.items()}
            fov = {nid: n.fov for nid, n in nodes.items()}
            Rs = {nid: n.R for nid, n in nodes.items()}
            # TODO: parametrize these
            H_default = np.logspace(1, 3, 1000)[0]
            k = -0.1
            safe_dist = 10
            connect_dist = 25
            _, sq = energyCoverage(self.adjacency_matrix(), current_coords, fov, Rs,
                                    H_default, k, safe_dist, connect_dist, DEFAULT_BBOX)
            self.surveillance_quality[i] = sq

        """
        Run Consensus
        """
        for l in range(L):
            neighbor_weights = {}
            neighbor_omegas = {}
            neighbor_qs = {}

            for id, n in nodes.items():
                weights = []
                omegas = []
                qs = []
                n_weights = nx.get_node_attributes(self.network,
                                                   'weights')[id]
                for neighbor in self.network.neighbors(id):
                    n_node = nx.get_node_attributes(self.network,
                                                    'node')[neighbor]
                    weights.append(n_weights[neighbor])
                    omegas.append(n_node.omega)
                    qs.append(n_node.qs)
                neighbor_weights[id] = weights
                neighbor_omegas[id] = omegas
                neighbor_qs[id] = qs

            for id, n in nodes.items():
                n.consensus_filter(neighbor_omegas[id],
                                   neighbor_qs[id],
                                   neighbor_weights[id])

        for id, n in nodes.items():
            n.intermediate_cov_update()

        """
        After Consensus Update
        """
        for id, n in nodes.items():
            n.after_consensus_update(len(nodes))

        for id, n in nodes.items():
            n.update_trackers(i, pre_consensus=False)

        trace_covs = self.get_trace_covariances()
        self.max_trace_cov[i] = max(trace_covs)
        self.mean_trace_cov[i] = np.mean(trace_covs)
        self.errors[i] = self.calc_errors(self.targets)
        self.adjacencies[i] = self.adjacency_matrix()
        self.weighted_adjacencies[i] = self.weighted_adjacency_matrix()

    def apply_failure(self, i, fail=None, mult=1, single_node_fail=False):
        nodes = nx.get_node_attributes(self.network, 'node')

        # Generate new R
        if fail is None:
            if single_node_fail:
                fail_node = 0
            else:
                fail_node = np.random.choice(list(nodes.keys()))

            # Get R from Node
            R = nodes[fail_node].R

            r_mat_size = R.shape[0]
            r = scipy.random.rand(r_mat_size, r_mat_size) * mult
            rpd = np.dot(r, r.T)
        else:
            fail_node = fail[0]
            rpd = fail[1]
            R = nodes[fail_node].R

        R = R + rpd
        nodes[fail_node].R = R
        self.failures[i] = (fail_node, rpd)
        return fail_node


    """
    Optimization  
    """
    def do_agent_opt(self, failed_node):
        nodes = nx.get_node_attributes(self.network, 'node')
        current_weights = nx.get_node_attributes(self.network, 'weights')

        cov_data = [n.omega for id, n in nodes.items()]

        covariance_data = []
        for c in cov_data:
            trace_c = np.trace(c)
            covariance_data.append(trace_c)

        new_config, new_weights = agent_opt(self.adjacency_matrix(),
                                            current_weights,
                                            covariance_data,
                                            failed_node=failed_node)

        G = nx.from_numpy_matrix(new_config)
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_team_opt(self, failed_node):
        nodes = nx.get_node_attributes(self.network, 'node')
        current_weights = nx.get_node_attributes(self.network, 'weights')

        cov_data = [n.full_cov_prediction for id, n in nodes.items()]
        omega_data = [n.omega for id, n in nodes.items()]

        # new_config, new_weights = team_opt(self.adjacency_matrix(),
        #                                     current_weights,
        #                                     cov_data,
        #                                     omega_data)
        new_config, new_weights = team_opt_bnb(self.adjacency_matrix(),
                                                current_weights,
                                                cov_data,
                                                omega_data,
                                                failed_node)
        # new_config, new_weights = team_opt_matlab(self.adjacency_matrix(),
        #                                     current_weights,
        #                                     cov_data,
        #                                     omega_data)
        G = nx.from_numpy_matrix(new_config)
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_greedy_opt(self, failed_node):
        nodes = nx.get_node_attributes(self.network, 'node')

        current_neighbors = list(self.network.neighbors(failed_node))

        cov_data = [n.full_cov_prediction for id, n in nodes.items()]

        best_cov_id = None
        best_cov = np.inf
        for neighbor_id in list(nodes):
            if neighbor_id not in current_neighbors:
                if np.linalg.det(cov_data[neighbor_id]) < best_cov:
                    best_cov_id = neighbor_id

        if best_cov_id is None:
            pass
        else:
            new_config = self.adjacency_matrix()
            new_config[failed_node, best_cov_id] = 1
            new_config[best_cov_id, failed_node] = 1

            G = nx.from_numpy_matrix(new_config)
            self.network = G
            nx.set_node_attributes(self.network, nodes, 'node')

            new_weights = self.get_metro_weights()
            nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_random_opt(self, failed_node):
        nodes = nx.get_node_attributes(self.network, 'node')
        current_neighbors = list(self.network.neighbors(failed_node))

        non_neighbors = []
        for neighbor_id in list(nodes):
            if neighbor_id not in current_neighbors:
                non_neighbors.append(neighbor_id)

        if len(non_neighbors) == 0:
            pass
        else:
            new_neighbor_id = np.random.choice(non_neighbors)

            new_config = self.adjacency_matrix()
            new_config[failed_node, new_neighbor_id] = 1
            new_config[new_neighbor_id, failed_node] = 1

            G = nx.from_numpy_matrix(new_config)
            self.network = G
            nx.set_node_attributes(self.network, nodes, 'node')

            new_weights = self.get_metro_weights()
            nx.set_node_attributes(self.network, new_weights, 'weights')

    """
    Network Operations
    """

    def adjacency_matrix(self):
        G = self.network
        num_nodes = len(list(G.nodes()))

        A = nx.adjacency_matrix(G).todense()
        if not np.array_equal(np.diag(A), np.ones(num_nodes)):
            A = A + np.diag(np.ones(num_nodes))
        return A

    def weighted_adjacency_matrix(self):
        A = self.adjacency_matrix()
        G = self.network

        for n in list(G.nodes()):
            weights = nx.get_node_attributes(G, 'weights')
            A[n, n] = weights[n][n]
            for i, neighbor in enumerate(list(G.neighbors(n))):
                A[n, neighbor] = weights[n][neighbor]

        return A

    def get_metro_weights(self):
        G = self.network
        num_nodes = len(list(self.network.nodes))
        weight_attrs = {}

        for i in range(num_nodes):
            weight_attrs[i] = {}
            self_degree = G.degree(i)
            metropolis_weights = []
            for n in G.neighbors(i):
                degree = G.degree(n)
                mw = 1 / (1 + max(self_degree, degree))
                weight_attrs[i][n] = mw
                metropolis_weights.append(mw)
            weight_attrs[i][i] = 1 - sum(metropolis_weights)

        return weight_attrs


    """
    Metric Calculations
    """

    def get_trace_covariances(self):
        nodes = nx.get_node_attributes(self.network, 'node')

        cov_data = []
        for id, node in nodes.items():
            cov_data.append(np.trace(node.full_cov))

        return cov_data

    def calc_errors(self, true_targets):
        nodes = nx.get_node_attributes(self.network, 'node')

        node_errors = {}
        for id, node in nodes.items():
            all_states = None
            for i, t in enumerate(true_targets):
                all_states = true_targets[i].state if all_states is None else \
                    np.concatenate((all_states, true_targets[i].state))

            errors = np.linalg.norm(node.full_state - all_states)
            node_errors[id] = errors

        return node_errors


def check_oob(state, ins):
    x = state[0][0]
    y = state[1][0]

    x_out_of_bounds = x < (DEFAULT_BBOX[0][0] + 5) or x > (DEFAULT_BBOX[0][1] - 5)
    if x_out_of_bounds:
        if x < DEFAULT_BBOX[0][0]:
            ins[0][0] = abs(state[2][0])
        else:
            ins[0][0] = -1 * state[2][0]
    y_out_of_bounds = y < (DEFAULT_BBOX[1][0] + 5) or y > (DEFAULT_BBOX[1][1] - 5)
    if y_out_of_bounds:
        if y < DEFAULT_BBOX[0][0]:
            ins[1][0] = abs(state[3][0])
        else:
            ins[1][0] = -1 * state[3][0]
    return ins