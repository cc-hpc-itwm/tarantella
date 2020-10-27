

# install clang-7.1.0
yum install llvm-toolset-7

module load cmake
module load curl 
spack install llvm@7.1.0 -gold -polly -python -compiler-rt %gcc@8.3.0  
spack install binutils -gold %clang@7.1.0

# install boost compiled with clang
spack install boost@1.67.0 -thread -signals -serialization -regex -random +pic -locale -log -graph -filesystem -chrono -atomic cxxstd=11 %clang@7.1.0


spack install automake %gcc@8.3.0
spack install m4 %gcc@8.3.0
spack install libtool %gcc@8.3.0
spack install autoconf %gcc@8.3.0

spack find -l autoconf
# ==> 1 installed package
# -- linux-centos7-x86_64 / gcc@8.3.0 -----------------------------
# zuqiblm autoconf@2.69

spack find -l libtool
# ==> 1 installed package
# -- linux-centos7-x86_64 / gcc@8.3.0 -----------------------------
# r7ax6ye libtool@2.4.6

# install GPI-2 for clang
module load binutils/2.33.1-clang-7.1.0-kgp5l 
source scl_source enable llvm-toolset-7

module load gcc/8.3.0-gcc-7.3.1-tsrcq 
module load llvm/7.1.0-gcc-8.3.0-3zpnd 
module load autoconf/2.69-gcc-8.3.0-zuqib 
module load automake/1.16.1-gcc-8.3.0-6i3ik 
module load binutils/2.33.1-clang-7.1.0-kgp5l 
module load libtool/2.4.6-gcc-8.3.0-r7ax6 
module load libsigsegv/2.12-gcc-8.3.0-cyvhh 
module load m4/1.4.18-gcc-8.3.0-zhwte 
spack install gpi-2@1.4.0 +ethernet +fPIC %clang@7.1.0 ^/r7ax6ye ^/zhwtegh ^/zuqiblm



