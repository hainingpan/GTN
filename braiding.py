import numpy as np
from GTN import *
class Braiding:

    def __init__(self, L, p, left_pos=None, right_pos=None,k=1,):
        '''p should > 0.5 in order to initialize a "trivial-topological-trivial" junction '''
        self.L = L
        self.p = p
        self.site_lists = []
        self.p_lists = []
        self.step_number= []
        self.left_pos = L // 4 if left_pos is None else left_pos
        self.right_pos = 3 * L // 4 if right_pos is None else right_pos
        self.k=k
        self.initialize_index()

    def initialize_index(self,k=None):
        '''pbp: stands for p (1-p) p, where b is p *inverse*'''
        if k is not None:
            k0=self.k
            self.k=k

        i_list = np.arange(0, 2 * self.L, 2)

        self.site_0_list = [[[leg * 2 * self.L + i, leg * 2 * self.L + i + 1]
                             for i in i_list] for leg in range(3)]
        self.site_1_list = [[[
            leg * 2 * self.L + i + 1, leg * 2 * self.L + i + 2
        ] for i in i_list[:-1]] for leg in range(3)]
        self.p_pbp = list(
            interpolation(x1=self.left_pos,
                          x2=self.right_pos,
                          l0=self.p,
                          h0=1 - self.p,
                          L=self.L,
                          k=self.k))
        self.p_bpb = list(
            interpolation(x1=self.left_pos,
                          x2=self.right_pos,
                          l0=1 - self.p,
                          h0=self.p,
                          L=self.L - 1,
                          k=self.k))
        self.p_pb = list(interpolation(x1=self.left_pos, x2=self.L+2, l0=self.p, h0=1-self.p, L=self.L,k=self.k))
        self.p_bp = list(interpolation(x1=self.left_pos, x2=self.L+2, l0=1-self.p, h0=self.p, L=self.L-1,k=self.k))
        self.p_p = [self.p] * self.L
        self.p_b = [1 - self.p] * (self.L - 1)
        self.outer = {leg: (leg) * 2 * self.L for leg in range(3)}
        self.inner = {leg: (1 + leg) * 2 * self.L - 1 for leg in range(3)}

        if k is not None:
            self.k=k0



    def initialization(self, time=5,k=None):
        if k is not None:
            self.initialize_index(k=k)
        
        p_0_list = self.p_pbp + self.p_pbp + self.p_p
        p_1_list = self.p_bpb + self.p_bpb + self.p_b
        site_0_list_all = sum(self.site_0_list, [])
        site_1_list_all = sum(self.site_1_list, [])
        site_list = [site_0_list_all, site_1_list_all] * time
        p_list = [p_0_list, p_1_list] * time
        self.initialize_index(k=self.k)
        return site_list, p_list

    def initialization_op(self, time=5):
        p_0_list=self.p_pb+self.p_pb+self.p_p 
        p_1_list=self.p_bp+self.p_bp+self.p_b + [self.p]
        site_0_list_all = sum(self.site_0_list, [])
        site_1_list_all = sum(self.site_1_list, []) + [[self.inner[0],self.inner[1]]]
        site_list = [site_0_list_all, site_1_list_all] * time
        p_list = [p_0_list, p_1_list] * time
        return site_list, p_list

    def shift_DW(self, leg, start, end, time=10, drop_0=None, drop_1=None):
        '''
        time: total time step of shifting of domain wall from `start` to `end`
        leg: which leg is operate
        start: [left, right] the position of left and right domain wall in the starting stage
        end: [left, right] the position of left and right domain wall in the final stage
        drop_0: [left, right] drop idx<=left and idx>=right for 0-indexed
        drop_1: [left, right] drop idx<=left and idx>=right for 1-indexed
        The evolution trajectory is the interpolation between the start and end.
        '''
        assert leg in range(3), f'leg number {leg} should be 0,1,2'
        if drop_0 is None:
            drop_0 = [-1, self.L]
        if drop_1 is None:
            drop_1 = [-1, self.L - 1]

        # start_left_pos,start_right_pos=start, start
        # end_left_pos, end_right_pos=end,end
        intermediate_pos = np.linspace(start, end, time).tolist()
        # intermediate_1_pos=np.linspace(start,end,time).tolist()
        p_list = []
        for intermediate in intermediate_pos:
            p_list.append(
                interpolation(x1=intermediate[0],
                              x2=intermediate[1],
                              l0=self.p,
                              h0=1 - self.p,
                              L=self.L,
                              k=self.k)[drop_0[0] + 1:drop_0[1]])
            p_list.append(
                interpolation(x1=intermediate[0],
                              x2=intermediate[1],
                              l0=1 - self.p,
                              h0=self.p,
                              L=self.L - 1,
                              k=self.k)[drop_1[0] + 1:drop_1[1]])

        site_list = [
            self.site_0_list[leg][drop_0[0] + 1:drop_0[1]],
            self.site_1_list[leg][drop_1[0] + 1:drop_1[1]]
        ] * time

        return site_list, p_list

    def link(self, left, right, time=4):

        site_list = [[[self.inner[left], self.inner[right]]]] * time
        p_list = [[self.p]] * time
        return site_list, p_list

    def pipeline(self, steps):
        step_number=0
        for step_number,(step, kwargs) in enumerate(steps):
            site_list, p_list = step(**kwargs)
            self.site_lists += site_list
            self.p_lists += p_list
            self.step_number+=[step_number]*len(site_list)
            
