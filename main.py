import argparse
import numpy as np
from utilities import *
 
parser = argparse.ArgumentParser()
parser.add_argument('-m',type=int,default=4,help="Mass of the particle")
parser.add_argument('--tau','-t',type=float,required = True, help="The time step at which to evaluate the density matrix")
parser.add_argument('--ord',type=int,default=40,help='The number of orders of partial waves to consider')
parser.add_argument('--err',type=float,default=1e-3,help="The reconstruction error in the new variables")
parser.add_argument('--dim',type=int,default=2,help="The dimension of the the simulation")
parser.add_argument('--dr',type=float,default=0.04,help="Radial grid spacing")
parser.add_argument('--box',type=float,nargs='+',default=None,help="Side lengths of the rectangular box, one per dimension")
parser.add_argument('-n',type=int,default=2,help="Order of the (q,s,z) expansion")
parser.add_argument('--n_d',type=int,default=2,help="Number of matrix squarings")
parsed_args = vars(parser.parse_args())
 
lam = 24.24/parsed_args["m"]
tau = parsed_args["tau"]
dim = parsed_args["dim"]
err = parsed_args["err"]
order = parsed_args["ord"]
n = parsed_args["n"]
 
lam_rel = 2*lam
 
rs = np.arange(0.5, 5.0, 0.001)
r_min = rs[np.argmax(np.array([tau*Aziz_potential(x) for x in rs]) < 30)]
if parsed_args["box"] is not None:
    box = np.array(parsed_args["box"])
    assert len(box) == dim, "--box needs one side length per dimension"
    r_max = np.sqrt(np.sum((box/2)**2)) + 3*np.sqrt(2*lam_rel*tau)
grid = np.arange(r_min, r_max, parsed_args["dr"])

#n_d = max(0, int(np.floor(np.log2(2*lam_rel*tau/(3*parsed_args["dr"])**2))))
if parsed_args["n_d"] is None:
    n_d = max(0, int(np.floor(np.log2(2*lam_rel*tau/parsed_args["dr"]**2))))
else:
    n_d = parsed_args["n_d"]
def write_tables(q, coeff, ud, tag):
    hdr = f"tau {tau} lam_rel {lam_rel} n {n} dim {dim}"
    np.savetxt(f"ukj{tag}.dat", np.column_stack([q, coeff]), header=hdr)
    np.savetxt(f"ud{tag}.dat", np.column_stack([grid, ud]), header=hdr)
 
def calc_pair_action(lam,tau,err,order):
    cache = {}
    res = {}
    for func, tag in [(pair_action, ""), (pair_action_dtau, "_dtau"), (pair_action_dlam, "_dlam")]:
        q, coeff, ud, resid = fit_ukj(grid,lam,tau,n_d,order,n,func,dim,cache)
        write_tables(q, coeff, ud, tag)
        print(f"u{tag}: max fit residual {np.max(resid):.2e}")
        res[tag] = np.max(resid)
    return res[""] < err
 
def main():
    print(f"dim {dim}, tau {tau}, lam_rel {lam_rel}, grid [{grid[0]:.3f}, {grid[-1]:.3f}] N={len(grid)}, n_d {n_d}, ord {order}, n {n}")
    if dim in [1,2,3]:
        stat = calc_pair_action(lam_rel,tau,err,order)      
    else:
        print("Dimension ",dim," not supported")
        stat = False
    if(stat):
        print("Pair action succesfully constructed")
    else:
        print("Pair action fit residual above --err; raise -n or narrow the grid")
 
if __name__ == '__main__':
    main()

