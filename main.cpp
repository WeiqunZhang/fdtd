#include "adi.H"
#include <AMReX.H>

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {
        ADI adi;
        adi.initData();
        adi.evolve();
    }
    amrex::Finalize();
}
