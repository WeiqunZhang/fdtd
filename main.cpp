#include "fdtd.H"
#include <AMReX.H>

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {
        FDTD fdtd;
        fdtd.initData();
        fdtd.evolve();
    }
    amrex::Finalize();
}
