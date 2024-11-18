from sympy import *
from sympy.physics.quantum import *
from tqdm import tqdm

## Recursive apply
def recursiveapply(expr,subs):
    expr1=expr.subs(subs)
    counter=0
    while not expr1== expr:
        expr=expr1
        expr1=expr.subs(subs).expand()
        counter+=1
        # print(expr1)
    return expr1
## Variables
g={i:Operator('\gamma_{}'.format(i)) for i in range(8)}
eta={i:Operator('\eta_{}'.format(i)) for i in range(8)}
xi={i:Operator(r'\xi_{}'.format(i)) for i in range(8)}

c={label:Operator(f'c_{label}') for label in ['xi','eta','g']+list(range(4))}
## Rules
external_leg=[(x,0) for x in list(g.values())+list(eta.values())]

constant_only=[(x,0) for x in list(xi.values())+list(eta.values())]
constant_gamma_only=[(x,0) for x in g.values()]
### normal order 
selfadjoint=[(Dagger(gi),gi) for gi in list(g.values())+list(eta.values())+list(xi.values())]
normalorder=[gi[i] for gi in [xi,eta,g] for i in range(len(g))]
normalorder2=[(normalorder[j]*normalorder[i],-normalorder[i]*normalorder[j])  if j>i else (normalorder[j]*normalorder[i],1) for i in range(len(normalorder)) for j in range(i,len(normalorder))]

normalorder_C=[element for pair in zip([Dagger(c[i]) for i in range(4)], [c[i] for i in range(4)]) for element in pair]

normalorder_C_dag=[Dagger(c[i]) for i in range(4)]
normalorder_C=[(c[i]) for i in range(4)]

normalorder2_C=[(normalorder_C_dag[i]*normalorder_C_dag[j],-normalorder_C_dag[j]*normalorder_C_dag[i],) if j>i else (normalorder_C_dag[i]*normalorder_C_dag[j],0) for i in range(len(normalorder_C_dag)) for j in range(i,len(normalorder_C_dag))]+[(normalorder_C[i]*normalorder_C[j],-normalorder_C[j]*normalorder_C[i]) if j>i else (normalorder_C[i]*normalorder_C[j],0) for i in range(len(normalorder_C)) for j in range(i,len(normalorder_C))]+[(normalorder_C[i]*normalorder_C_dag[j],-normalorder_C_dag[j]*normalorder_C[i]) if j!=i else (normalorder_C[i]*normalorder_C_dag[j],1-normalorder_C_dag[j]*normalorder_C[i]) for i in range(len(normalorder_C)) for j in range(len(normalorder_C))]

c2g=[(c['xi'],(xi[0]-I*xi[1])/2),
(c['eta'],(eta[0]-I*eta[1])/2),
(c['g'],(g[0]-I*g[1])/2),
(Dagger(c['xi']),(xi[0]+I*xi[1])/2),
(Dagger(c['eta']),(eta[0]+I*eta[1])/2),
(Dagger(c['g']),(g[0]+I*g[1])/2),
(c[0],(g[0]-I*g[1])/2),
(c[1],(g[2]-I*g[3])/2),
(c[2],(g[4]-I*g[5])/2),
(c[3],(g[6]-I*g[7])/2),
]

gamma_xi=[(g[idx],xi[idx]) for idx in range(8)]

rho_EPR0=lambda s0,s1: (1+I*s0*s1)/2


def get_Gamma(op_dm,dim,normalize=True):
    Gamma = zeros(2*dim, 2*dim)
    order=[eta[i] for i in range(dim)] + [xi[i] for i in range(dim)]
    ij_list=[(i,j) for i in range(2*dim) for j in range(i+1,2*dim)]

    # op_K=op_u_g.subs(gamma_xi)
    # op_dm=recursiveapply(( op_K* rho_epr4 * Dagger(op_K)).expand().subs(selfadjoint),normalorder2)
    for i,j in tqdm(ij_list):
            rs=I*order[i]*order[j]*( op_dm)
            matel=recursiveapply(rs.expand(),normalorder2).subs(constant_only)*16
            matel=matel.simplify()
            Gamma[i,j]=matel
            Gamma[j,i]=-matel
    if normalize:
        norm=np.sqrt(-(Gamma@Gamma)[0,0])
        Gamma=Gamma/norm
    return Gamma

def contraction_(Gamma,Upsilon,i):
    Gamma_LL=Gamma[:i,:i]
    Gamma_LR=Gamma[:i,i:]
    Gamma_RR=Gamma[i:,i:]
    Upsilon_LL=Upsilon[:i,:i]
    Upsilon_RR=Upsilon[i:,i:]
    Upsilon_RL=Upsilon[i:,:i]
    Mat1=Matrix([[Gamma_LL,0],[0,Gamma_RR]])
    
    pos_mat={}
    for i in range(2):
        for j in range(2):
            mat_zero=zeros(2)
            mat_zero[i,j]=1
            pos_mat[(i,j)]=mat_zero
    one=eye(Gamma_LL.cols)

    mat1=KroneckerProduct(pos_mat[(0,0)],Gamma_LL)+KroneckerProduct(pos_mat[(1,1)],Upsilon_RR)

    mat2=KroneckerProduct(pos_mat[(0,0)],Gamma_LR)+KroneckerProduct(pos_mat[(1,1)],Upsilon_RL)
    mat3=KroneckerProduct(pos_mat[(0,0)],Gamma_RR)+KroneckerProduct(pos_mat[(1,1)],Upsilon_LL)+KroneckerProduct(pos_mat[(0,1)],one)+KroneckerProduct(pos_mat[(1,0)],-one)
    # return mat1, mat2, mat3
    return mat1 + mat2 @ (mat3).inv() @ mat2.T


rho_epr4=rho_EPR0(eta[0],xi[0]) * rho_EPR0(eta[1],xi[1]) * rho_EPR0(eta[2],xi[2]) * rho_EPR0(eta[3],xi[3])
rho_epr2=rho_EPR0(eta[0],xi[0]) * rho_EPR0(eta[1],xi[1]) 
rho_epr6=rho_EPR0(eta[0],xi[0]) * rho_EPR0(eta[1],xi[1]) * rho_EPR0(eta[2],xi[2]) * rho_EPR0(eta[3],xi[3]) * rho_EPR0(eta[4],xi[4]) * rho_EPR0(eta[5],xi[5]) 
rho_epr8=rho_EPR0(eta[0],xi[0]) * rho_EPR0(eta[1],xi[1]) * rho_EPR0(eta[2],xi[2]) * rho_EPR0(eta[3],xi[3]) * rho_EPR0(eta[4],xi[4]) * rho_EPR0(eta[5],xi[5]) * rho_EPR0(eta[6],xi[6]) * rho_EPR0(eta[7],xi[7])
