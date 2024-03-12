check rv_continuous to replace the random generator
check Chaoming


1. use same PRB convention
2. jump does not make sense, why?
3. swap a1 and a2
4. MI move by 1 (not much change)


very large ensemble for forced measurement

why long range for , track how it changes, produce a movie; (done it shows certain long range )

antisymmetry check / Gamma^2+1=0 check (does not work)

n2,n3 is not uniformed; (yes)

could it be mislabeling of parameters? sweep a1 to see if MI can be as small as 0.1, if not, whether change the rv_continuous solves it

---
what questions to ask

problem:
1. criticality for Born is different
2. MI is diff, mine is much larger than yours in the critical phase
3. cannot find arealaw in forced measurement
   
Questions:
1. what convention of sign?
2. what parameters
3. ensemble system size, 
4. evolution time, how long and did you take average over different t
5. It seems that the state can jump from area-law back to critical phase 
6. initial state


---
Steps:
tensor network -> gate operator/covariance matrix  (seems to be correct from reproduction of Anna's)

use the correct circuit (i.e., probability)

correct parameters (ensemble size, time step..)

correct mutual information (but even it is wrong, the criticality shouldn't be affected)

why does it have a different region in forced measurement (shouldn't more unitary close to critical phase)
----

1. sqrt(1-p) exp(I g_i g_i+1) * Im phi, p D_+, pD_-}

Born vs forced
1. No domanin wall, find area law phase, find trans pt of p
2. domain-wall, EC, tune p
3. tri-junction 

---

2023/07/24
Task 1:
figure out the braiding protocol, using tri-junction
measure the mutual information etc to show the braiding statistics
this is in class DIII

Task 2:
Consider class A, the unitary is U(1,1)
it seems that there is a U^dag sigam^z U = sigma^z, where U = exp(i alpha * sigma^z + beta * sigma^x + gamma * sigam^y + i theta * id)

- [ ] Verify the transfromation matrix, write in the 2nd quantized form. find the Kraus op set
- [ ] Check QA-QB conservation. Here, Qa and Qb are just number density
- [ ] Find the area phase by parameterization, maybe beta is a starting point
- [ ] Find domain wall

---
- [x] add a illustrator for Fig. 1


(Hi Chao-Ming, I have produced .. and also updated the overleaf which we had before [link], please ...)

-[ ] Entanglement entropy precision with problem if 6L~384


$\exp(\alpha (c_1^\dagger c_2+ c_2^\dagger c_1) + \beta (i c_1^\dagger c_2-i c_2^\dagger c_1)+i\gamma(c_1^\dagger c_1-c_2^\dagger c_2))\exp(i\theta)$