import argparse
import numpy as np
from utilities import *
import configparser 
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-m',type=int,default=4,help="Mass of the particle")
parser.add_argument('--tau','-t',type=float,required = True, help="The time step at which to evaluate the density matrix")
parser.add_argument('--ord',type=int,default=40,help='The number of orders of partial waves to consider')
parser.add_argument('--err',type=float,default=1e-3,help="The reconstruction error in the new variables")
parser.add_argument('--dim',type=int,default=2,help="The dimension of the the simulation")
parser.add_argument('--dr',type=float,default=0.04,help="Radial grid spacing")
parser.add_argument('--box',type=float,nargs='+',default=None,help="Side lengths of the rectangular box, one per dimension")
parser.add_argument('-n',type=int,default=2,help="Order of the (q,s,z) expansion")
parser.add_argument('--n_tau',type=int,default=2,help="Order of the (q,s,z) expansion for the tau derivative")
parser.add_argument('--n_d',type=int,default=None,help="Number of matrix squarings")
parser.add_argument('--determine', action=argparse.BooleanOptionalAction, default=False,help="Determine the optimal parameters, can take a long time depending on the box size")
parser.add_argument('--suffix', type=str, default='',help="Suffix to append to the metadata and data folder")
parsed_args = vars(parser.parse_args())
 
lam = 24.24/parsed_args["m"]
tau = parsed_args["tau"]
dim = parsed_args["dim"]
err = parsed_args["err"]
order = parsed_args["ord"]
n = parsed_args["n"]
n_t = parsed_args["n_tau"]
dr = parsed_args["dr"]
suffix = parsed_args["suffix"]

lam_rel = 2*lam

#Specify the potential function 
pot_f = potential.aziz_1995

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
def write_ini(path, sections):
    with open(path, 'w') as f:
        for name, params in sections.items():
            f.write(f'[{name}]\n')
            for key, val in params.items():
                if isinstance(val, (list, tuple, np.ndarray)):
                    for v in val:
                        f.write(f'{key} = {v}\n')
                else:
                    f.write(f'{key} = {val}\n')
            f.write('\n')

def write_metadata(lam_rel,tau,err,order,n_d,grid):

    # 1. Initialize the config parser
    write_ini(f'config_{suffix}.ini', {
        'Simulation Parameters': {
            'potential' :pot_f.__name__,
            'dim': dim,
            'tau': tau,
            'box': [f'{b:.3f}' for b in box],
            'lambda_rel': lam_rel,
            'n': n,
            'n_tau': n_t,
            'grid': [f'{grid[0]:.3f}', f'{grid[-1]:.3f}'],
        },
        'Generation Parameters': {
            'order': order,
            'squarings': n_d,
            'dr': dr,
        },
        'Data': {
            'parent': 'Pair_data_' + suffix,
        },
    })

def write_tables(q, coeff, ud, tag):
    ukgpath = Path(f"Pair_data_{suffix}/ukj{tag}.dat")
    udpath = Path(f"Pair_data_{suffix}/ud{tag}.dat")

    ukgpath.parent.mkdir(parents=True, exist_ok=True)
    udpath.parent.mkdir(parents=True, exist_ok=True)
    hdr = f"tau {tau} lam_rel {lam_rel} n {n} dim {dim}"
    np.savetxt(ukgpath, np.column_stack([q, coeff]), header=hdr)
    np.savetxt(udpath, np.column_stack([grid, ud]), header=hdr)
 
def calc_pair_action(lam,tau,err,order,n_d,grid):
    cache = {}
    res = {}
    write_metadata(lam,tau,err,order,n_d,grid)
    for func, tag,n_f in [(pair_action, "",n), (pair_action_dtau, "_dtau",n_t), (pair_action_dlam, "_dlam",n)]:
        q, coeff, ud, resid = fit_ukj(grid,lam,tau,n_d,order,n_f,func,dim,cache,pot_f)
        write_tables(q, coeff, ud, tag)
        print(f"u{tag}: max fit residual {np.max(resid):.2e}")
        res[tag] = np.max(resid)
    return res[""] < err

def main():
    n_d_run, order_run, dr_run = n_d, order, dr   # start from the CLI/global values
    grid_run = np.arange(r_min, r_max, dr_run)

    print("Calculating simulation with parameters:\n")
    print(f"dim {dim}, tau {tau}, lam_rel {lam_rel}, grid [{grid_run[0]:.3f}, {grid_run[-1]:.3f}] "
          f"N={len(grid_run)}, n_d {n_d_run}, ord {order_run}, n {n}")
    if dim in [1, 2, 3]:
       stat = calc_pair_action(lam_rel, tau, err, order_run, n_d_run, grid_run)
    else:
        print("Dimension ", dim, " not supported")
        stat = False
    if(stat):
        print("Pair action succesfully constructed")
    else:
        print("Pair action fit residual above --err; raise -n or narrow the grid")
 
if __name__ == '__main__':
    main()

