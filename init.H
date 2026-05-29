#ifndef INIT_H_
#define INIT_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

void InitSetupFields(
    std::string const &pp_prefix,
    std::string const &ic,
    amrex::Real sinwave_amplitude,
    int sinwave_dir,
    int sinwave_pol,
    amrex::Real sinwave_wavelength,
    amrex::Geometry const &geom,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &efields,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &bfields);

#endif
