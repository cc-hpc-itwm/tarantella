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
#     spack install onednn    
#
# You can edit this file again by typing:
#
#     spack edit onednn    
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *


class Onednn(CMakePackage):
    """Install one-DNN"""

    homepage = "https://github.com/oneapi-src/oneDNN.git"
    git      = "https://github.com/oneapi-src/oneDNN.git"

    version('1.4.0', tag='v1.4')

    depends_on('cmake@2.8.11:')

    variant('cpu_runtime',
            default='OMP',
            values=('OMP', 'TBB', 'SEQ', 'THREADPOOL'),
            multi=False,
            description='Defines the threading runtime for CPU engines.')


    def cmake_args(self):
        args = []
        if 'cpu_runtime' in self.spec:
            args.extend(['-DDNNL_CPU_RUNTIME=' % self.spec['cpu_runtime']
                        ])

        return args




