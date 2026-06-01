
#include "adi.H"
#include "init.H"
#include "util.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <cmath>

using namespace amrex;

namespace
{
    MultiFab makeRhsLike(MultiFab const &field)
    {
        return MultiFab(field.boxArray(), field.DistributionMap(), 1, 0);
    }
} // namespace

ADI::ADI()
{
    ParmParse pp("adi");
    pp.getarr("n_cells", m_n_cells);
    pp.query("max_grid_size", m_max_grid_size);

    RealVect prob_lo, prob_hi;
    pp.getarr("prob_lo", prob_lo);
    pp.getarr("prob_hi", prob_hi);

    pp.query("max_step", m_max_step);
    pp.query("plot_int", m_plot_int);
    pp.query("plot_format", m_plot_format);
    pp.query("cfl", m_cfl);
    pp.query("output_dir", m_output_dir);

    if (m_plot_format != "numpy" && m_plot_format != "visit")
    {
        amrex::Abort("adi.plot_format must be \"numpy\" or \"visit\"");
    }

    pp.query("ic", m_ic);
    pp.query("ic_amplitude", m_ic_amplitude);
    pp.query("ic_dir", m_ic_dir);
    pp.query("ic_pol", m_ic_pol);
    pp.query("ic_wavelength", m_ic_wavelength);

    m_pulse_center = 0.5_rt * (prob_lo[m_ic_dir] + prob_hi[m_ic_dir]);
    pp.query("pulse_center", m_pulse_center);
    pp.query("pulse_sigma", m_pulse_sigma);

    Box domain(IntVect(0), m_n_cells - 1);
    RealBox real_box(prob_lo.begin(), prob_hi.begin());
    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};

    m_geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    m_grids.define(domain);
    m_grids.maxSize(m_max_grid_size);

    m_dmap.define(m_grids);

    static_assert(AMREX_SPACEDIM == 3, "3D only");
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        IntVect etyp(1); // nodal by default
        etyp[idim] = 0;  // cell-centered in idim-direction
        m_efields[idim].define(amrex::convert(m_grids, etyp), m_dmap,
                               1, 1); // one component, one ghost
        IntVect btyp(0);              // cell-centerd by default
        btyp[idim] = 1;               // nodal in idim-direction
        m_bfields[idim].define(amrex::convert(m_grids, btyp), m_dmap, 1, 1);
    }
}

void ADI::initData()
{
    InitSetupFields("adi", m_ic, m_ic_amplitude, m_ic_dir,
                    m_ic_pol, m_ic_wavelength, m_pulse_center, m_pulse_sigma,
                    m_geom, m_efields, m_bfields);
}

void ADI::evolve()
{
    constexpr Real c = 2.99792458e8;

    auto dxinv = m_geom.InvCellSizeArray();
    Real inv_dx2_sum = 0.0_rt;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        inv_dx2_sum += dxinv[idim] * dxinv[idim];
    }
    Real dt = m_cfl / (c * std::sqrt(inv_dx2_sum));

    Real time = 0.0_rt;

    if (m_plot_int > 0)
    {
        UtilWritePlotOutput(m_plot_format, m_output_dir, 0, time,
                            m_grids, m_dmap, m_geom, m_efields, m_bfields);
    }

    for (int step = 0; step < m_max_step; ++step)
    {
        adiFirstHalfStep(dt);
        adiSecondHalfStep(dt);

        time += dt;

        if (m_plot_int > 0 && (step + 1) % m_plot_int == 0)
        {
            UtilWritePlotOutput(m_plot_format, m_output_dir, step + 1, time,
                                m_grids, m_dmap, m_geom, m_efields, m_bfields);
        }
    }
}

void ADI::adiFirstHalfStep(Real dt)
{
    // eq:adi-first-half-amrex — implicit E along y,z,x; explicit B at n+1/2
    auto const period = m_geom.periodicity();
    Vector<MultiFab *> efields{AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])};
    Vector<MultiFab *> bfields{AMREX_D_DECL(&m_bfields[0], &m_bfields[1], &m_bfields[2])};

    amrex::FillBoundary(efields, period);
    amrex::FillBoundary(bfields, period);

    MultiFab rhs_ex = buildRhsEx1(dt);
    solveImplicitEx1(m_efields[0], rhs_ex, dt);

    MultiFab rhs_ey = buildRhsEy1(dt);
    solveImplicitEy1(m_efields[1], rhs_ey, dt);

    MultiFab rhs_ez = buildRhsEz1(dt);
    solveImplicitEz1(m_efields[2], rhs_ez, dt);

    amrex::FillBoundary(efields, period);

    stepBx(dt);
    stepBy(dt);
    stepBz(dt);

    amrex::FillBoundary(bfields, period);
}

void ADI::adiSecondHalfStep(Real dt)
{
    // eq:adi-second-half-amrex — implicit E along z,x,y; explicit B at n+1
    auto const period = m_geom.periodicity();
    Vector<MultiFab *> efields{AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])};
    Vector<MultiFab *> bfields{AMREX_D_DECL(&m_bfields[0], &m_bfields[1], &m_bfields[2])};

    amrex::FillBoundary(efields, period);
    amrex::FillBoundary(bfields, period);

    MultiFab rhs_ex = buildRhsEx2(dt);
    solveImplicitEx2(m_efields[0], rhs_ex, dt);

    MultiFab rhs_ey = buildRhsEy2(dt);
    solveImplicitEy2(m_efields[1], rhs_ey, dt);

    MultiFab rhs_ez = buildRhsEz2(dt);
    solveImplicitEz2(m_efields[2], rhs_ez, dt);

    amrex::FillBoundary(efields, period);

    stepBx(dt);
    stepBy(dt);
    stepBz(dt);

    amrex::FillBoundary(bfields, period);
}

MultiFab ADI::buildRhsEx1(Real dt) const
{
    // RHS of eq:adi-first-half-amrex Ex row (vacuum), for tridiagonal solve along y.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(m_efields[0]);

    auto const dx = m_geom.CellSizeArray();
    Real const dy = dx[1];
    Real const dyinv = 1.0_rt / dy;
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];

    Real const coef_ex = 4.0_rt * dy * dy / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dy * dy / dt;
    Real const coef_ey = dy;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ex_arr = m_efields[0].const_array(mfi);
        auto const &ey_arr = m_efields[1].const_array(mfi);
        auto const &bz_arr = m_bfields[2].const_array(mfi);
        auto const &by_arr = m_bfields[1].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dyinv * (bz_arr(i, j, k) - bz_arr(i, j - 1, k)) -
                                dzinv * (by_arr(i, j, k) - by_arr(i, j, k - 1));
            Real const dey_dx = dxinv * ((ey_arr(i + 1, j - 1, k) - ey_arr(i, j - 1, k)) -
                                         (ey_arr(i + 1, j, k) - ey_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ex * ex_arr(i, j, k) + coef_b * curl_b + coef_ey * dey_dx;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy1(Real dt) const
{
    // RHS of eq:adi-first-half-amrex Ey row (vacuum), for tridiagonal solve along z.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(m_efields[1]);

    auto const dx = m_geom.CellSizeArray();
    Real const dz = dx[2];
    Real const dzinv = 1.0_rt / dz;
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];

    Real const coef_ey = 4.0_rt * dz * dz / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dz * dz / dt;
    Real const coef_ez = dz;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ey_arr = m_efields[1].const_array(mfi);
        auto const &ez_arr = m_efields[2].const_array(mfi);
        auto const &bx_arr = m_bfields[0].const_array(mfi);
        auto const &bz_arr = m_bfields[2].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dzinv * (bx_arr(i, j, k) - bx_arr(i, j, k - 1)) -
                                dxinv * (bz_arr(i, j, k) - bz_arr(i - 1, j, k));
            Real const dez_dy = dyinv * ((ez_arr(i, j + 1, k - 1) - ez_arr(i, j, k - 1)) -
                                         (ez_arr(i, j + 1, k) - ez_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ey * ey_arr(i, j, k) + coef_b * curl_b + coef_ez * dez_dy;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz1(Real dt) const
{
    // RHS of eq:adi-first-half-amrex Ez row (vacuum), for tridiagonal solve along x.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(m_efields[2]);

    auto const dx = m_geom.CellSizeArray();
    Real const dx_cell = dx[0];
    Real const dxinv = 1.0_rt / dx_cell;
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];

    Real const coef_ez = 4.0_rt * dx_cell * dx_cell / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dx_cell * dx_cell / dt;
    Real const coef_ex = dx_cell;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ez_arr = m_efields[2].const_array(mfi);
        auto const &ex_arr = m_efields[0].const_array(mfi);
        auto const &by_arr = m_bfields[1].const_array(mfi);
        auto const &bx_arr = m_bfields[0].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dxinv * (by_arr(i, j, k) - by_arr(i - 1, j, k)) -
                                dyinv * (bx_arr(i, j, k) - bx_arr(i, j - 1, k));
            Real const dex_dz = dzinv * ((ex_arr(i - 1, j, k + 1) - ex_arr(i - 1, j, k)) -
                                         (ex_arr(i, j, k + 1) - ex_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ez * ez_arr(i, j, k) + coef_b * curl_b + coef_ex * dex_dz;
        });
    }

    return rhs;
}

void ADI::solveImplicitEx1(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(ex, rhs, dt);
}

void ADI::solveImplicitEy1(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(ey, rhs, dt);
}

void ADI::solveImplicitEz1(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(ez, rhs, dt);
}

void ADI::stepBx(Real dt)
{
    // B_x += (dt/2)(dEy/dz - dEz/dy), vacuum Yee stencil (adi.tex magnetic update).
    auto const dxinv = m_geom.InvCellSizeArray();
    Real const halfdt = 0.5_rt * dt;

    auto const &ey = m_efields[1].arrays();
    auto const &ez = m_efields[2].arrays();
    auto const &bx = m_bfields[0].arrays();

    ParallelFor(m_bfields[0], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        bx[b](i, j, k) +=
            halfdt * (dxinv[2] * (ey[b](i, j, k + 1) - ey[b](i, j, k)) -
                      dxinv[1] * (ez[b](i, j + 1, k) - ez[b](i, j, k)));
    });
    Gpu::streamSynchronize();
}

void ADI::stepBy(Real dt)
{
    auto const dxinv = m_geom.InvCellSizeArray();
    Real const halfdt = 0.5_rt * dt;

    auto const &ex = m_efields[0].arrays();
    auto const &ez = m_efields[2].arrays();
    auto const &by = m_bfields[1].arrays();

    ParallelFor(m_bfields[1], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        by[b](i, j, k) +=
            halfdt * (dxinv[0] * (ez[b](i + 1, j, k) - ez[b](i, j, k)) -
                      dxinv[2] * (ex[b](i, j, k + 1) - ex[b](i, j, k)));
    });
    Gpu::streamSynchronize();
}

void ADI::stepBz(Real dt)
{
    auto const dxinv = m_geom.InvCellSizeArray();
    Real const halfdt = 0.5_rt * dt;

    auto const &ex = m_efields[0].arrays();
    auto const &ey = m_efields[1].arrays();
    auto const &bz = m_bfields[2].arrays();

    ParallelFor(m_bfields[2], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        bz[b](i, j, k) +=
            halfdt * (dxinv[1] * (ex[b](i, j + 1, k) - ex[b](i, j, k)) -
                      dxinv[0] * (ey[b](i + 1, j, k) - ey[b](i, j, k)));
    });
    Gpu::streamSynchronize();
}

MultiFab ADI::buildRhsEx2(Real dt) const
{
    // RHS of eq:adi-second-half-amrex Ex row (vacuum), for tridiagonal solve along z.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(m_efields[0]);

    auto const dx = m_geom.CellSizeArray();
    Real const dz = dx[2];
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dz;
    Real const dxinv = 1.0_rt / dx[0];

    Real const coef_ex = 4.0_rt * dz * dz / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dz * dz / dt;
    Real const coef_ez = dz;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ex_arr = m_efields[0].const_array(mfi);
        auto const &ez_arr = m_efields[2].const_array(mfi);
        auto const &bz_arr = m_bfields[2].const_array(mfi);
        auto const &by_arr = m_bfields[1].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dyinv * (bz_arr(i, j, k) - bz_arr(i, j - 1, k)) -
                                dzinv * (by_arr(i, j, k) - by_arr(i, j, k - 1));
            Real const dez_dx = dxinv * ((ez_arr(i + 1, j, k - 1) - ez_arr(i, j, k - 1)) -
                                         (ez_arr(i + 1, j, k) - ez_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ex * ex_arr(i, j, k) + coef_b * curl_b + coef_ez * dez_dx;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy2(Real dt) const
{
    // RHS of eq:adi-second-half-amrex Ey row (vacuum), for tridiagonal solve along x.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(m_efields[1]);

    auto const dx = m_geom.CellSizeArray();
    Real const dx_cell = dx[0];
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx_cell;
    Real const dyinv = 1.0_rt / dx[1];

    Real const coef_ey = 4.0_rt * dx_cell * dx_cell / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dx_cell * dx_cell / dt;
    Real const coef_ex = dx_cell;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ey_arr = m_efields[1].const_array(mfi);
        auto const &ex_arr = m_efields[0].const_array(mfi);
        auto const &bx_arr = m_bfields[0].const_array(mfi);
        auto const &bz_arr = m_bfields[2].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dzinv * (bx_arr(i, j, k) - bx_arr(i, j, k - 1)) -
                                dxinv * (bz_arr(i, j, k) - bz_arr(i - 1, j, k));
            Real const dex_dy = dyinv * ((ex_arr(i - 1, j + 1, k) - ex_arr(i - 1, j, k)) -
                                         (ex_arr(i, j + 1, k) - ex_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ey * ey_arr(i, j, k) + coef_b * curl_b + coef_ex * dex_dy;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz2(Real dt) const
{
    // RHS of eq:adi-second-half-amrex Ez row (vacuum), for tridiagonal solve along y.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(m_efields[2]);

    auto const dx = m_geom.CellSizeArray();
    Real const dy = dx[1];
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dy;
    Real const dzinv = 1.0_rt / dx[2];

    Real const coef_ez = 4.0_rt * dy * dy / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dy * dy / dt;
    Real const coef_ey = dy;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ez_arr = m_efields[2].const_array(mfi);
        auto const &ey_arr = m_efields[1].const_array(mfi);
        auto const &by_arr = m_bfields[1].const_array(mfi);
        auto const &bx_arr = m_bfields[0].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dxinv * (by_arr(i, j, k) - by_arr(i - 1, j, k)) -
                                dyinv * (bx_arr(i, j, k) - bx_arr(i, j - 1, k));
            Real const dey_dz = dzinv * ((ey_arr(i, j - 1, k + 1) - ey_arr(i, j - 1, k)) -
                                         (ey_arr(i, j, k + 1) - ey_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ez * ez_arr(i, j, k) + coef_b * curl_b + coef_ey * dey_dz;
        });
    }

    return rhs;
}

void ADI::solveImplicitEx2(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(ex, rhs, dt);
}

void ADI::solveImplicitEy2(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(ey, rhs, dt);
}

void ADI::solveImplicitEz2(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(ez, rhs, dt);
}
