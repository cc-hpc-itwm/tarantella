# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install gpi-2
#
# You can edit this file again by typing:
#
#     spack edit gpi-2
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *


class Gpi2(Package):
    """GPI-2."""

    homepage = "http://www.gpi-site.com"
    git      = "https://github.com/cc-hpc-itwm/GPI-2"

    version('2020-02-25', commit='bdc3d511b32481fd8dcc7357fdd1148f831b1a79')
    version('1.4.0', tag='v1.4.0')
    version('1.3.0', tag='v1.3.0')


    variant('ethernet', default=True,
            description="Ethernet support")
    variant('infiniband',  default=False,
            description="Infiniband support")
    variant('fortran',  default=False,
            description="Fortran support")
    variant('mpi',  default=False,
            description="enable mixed mode with MPI")
    variant('fPIC',  default=False,
            description="add `-fPIC` compiler flags")

    depends_on('autoconf', type='build')
    depends_on('automake', type='build')
    depends_on('libtool',  type='build')
    depends_on('m4',       type='build')

    def install(self, spec, prefix):
        install_opts = ['--prefix={0}'.format(prefix)]
        if '+ethernet' in spec:
                install_opts.append('--with-ethernet')
        else:
                install_opts.append('--without-ethernet')

        if '+infiniband' in spec:
                install_opts.append('--with-infiniband')
        else:
                install_opts.append('--without-infiniband')

        if '+fortran' in spec:
                install_opts.append('--with-fortran')
        else:
             	install_opts.append('--without-fortran')
        if '+mpi' in spec:
                install_opts.append('--with-mpi')
        else:
                install_opts.append('--without-mpi')

        flags = []
        if '+fPIC' in spec:
                flags.append('-fPIC')

        compile_flags = 'CFLAGS=\"' + ' '.join(flags) + '\" CPPFLAGS=\"' + ' '.join(flags) + '\"'
        autogen = Executable('./autogen.sh')
        autogen()
        conf = Executable('./configure ' + ' '.join(install_opts) + ' ' + compile_flags)
        conf()
        make()
        make('install')

