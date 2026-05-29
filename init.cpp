#include "init.H"

#include <AMReX_Utility.H>

#include <cmath>

using namespace amrex;

namespace
{
    constexpr Real c_light = 2.99792458e8;

    // Return the physical coordinate in one direction for a field component that is
    // staggered on a Yee grid: half-cell shifted along its own component direction.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Real
        staggered_coord(int component, int coord_dir, int i, int j, int k,
                        GpuArray<Real, AMREX_SPACEDIM> const &problo,
                        GpuArray<Real, AMREX_SPACEDIM> const &dx)
    {
        int idx = (coord_dir == 0) ? i : ((coord_dir == 1) ? j : k);
        Real offset = (component == coord_dir) ? 0.5_rt : 0.0_rt;
        return problo[coord_dir] + (idx + offset) * dx[coord_dir];
    }
} // namespace

void InitSetupFields(
    std::string const &pp_prefix,
    std::string const &ic,
    Real sinwave_amplitude,
    int sinwave_dir,
    int sinwave_pol,
    Real sinwave_wavelength,
    Geometry const &geom,
    Array<MultiFab, AMREX_SPACEDIM> &efields,
    Array<MultiFab, AMREX_SPACEDIM> &bfields)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        efields[idim].setVal(0);
        bfields[idim].setVal(0);
    }

    const bool is_sinwave = (ic == "sinwave");
    const bool is_standing_wave = (ic == "standingwave");
    const std::string ic_err = pp_prefix + ".ic must be 'sinwave' or 'standingwave'";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_sinwave || is_standing_wave, ic_err.c_str());

    AMREX_ALWAYS_ASSERT(sinwave_dir >= 0 && sinwave_dir < AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(sinwave_pol >= 0 && sinwave_pol < AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(sinwave_pol != sinwave_dir);

    auto problo = geom.ProbLoArray();
    auto probhi = geom.ProbHiArray();
    auto dx = geom.CellSizeArray();

    Real domain_len = probhi[sinwave_dir] - problo[sinwave_dir];
    Real wavelength = (sinwave_wavelength > 0) ? sinwave_wavelength : domain_len;
    Real kw = 2.0 * M_PI / wavelength;
    Real E0 = sinwave_amplitude;
    const int dir = sinwave_dir;
    const int pol = sinwave_pol;

    auto const &ea = efields[pol].arrays();
    // determine the direction of the magnetic field and its sign based on the right-hand rule
    const int bdir = 3 - dir - pol;
    const Real bsign = ((dir + 1) % 3 == pol) ? 1.0_rt : -1.0_rt;
    auto const &ba = bfields[bdir].arrays();

    ParallelFor(efields[pol], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        Real phase_coord = staggered_coord(pol, dir, i, j, k, problo, dx);
        Real s = std::sin(kw * phase_coord);
        ea[b](i,j,k) = E0 * s; });

    if (is_sinwave)
    {
        Real B0 = E0 / c_light;
        // B = (1/c) k_hat x E for a +dir traveling wave at t=0
        ParallelFor(bfields[bdir], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            Real phase_coord = staggered_coord(bdir, dir, i, j, k, problo, dx);
            Real s = std::sin(kw * phase_coord);
            ba[b](i,j,k) = bsign * B0 * s; });
    }
    // For a standing wave at t=0, initialize B to zero and only seed E.
    Gpu::streamSynchronize();

    Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&efields[0], &efields[1], &efields[2])};
    Vector<MultiFab *> bfield_ptrs{AMREX_D_DECL(&bfields[0], &bfields[1], &bfields[2])};
    amrex::FillBoundary(efield_ptrs, geom.periodicity());
    amrex::FillBoundary(bfield_ptrs, geom.periodicity());
}
