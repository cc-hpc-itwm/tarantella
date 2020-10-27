# Meant to be sourced from build jobs

function parseJobProperties {
    # Parses env variable CI_JOB_NAME which is expected as
    # "hpdlf:<COMPILER>:<COMPILER_VERSION>:<GPI_VERSION>:<BUILD_TYPE>[:<TF_ENV>]"
    # Examples: hpdlf:gcc:6.4.0:gpi-2/1.4.0:Debug:tf2.0, hpdlf:clang:4:gpi-2/1.4.0:Release
    errorMsg="Environment variable CI_JOB_NAME must be set to hpdlf:<COMPILER>:<COMPILER_VERSION>:<GPI_VERSION>:<BUILD_TYPE>[:<TF_ENV>]. \
Found: '${CI_JOB_NAME:-}'"
    if [ "${CI_JOB_NAME}" == "" ]; then
        echo "${errorMsg}" 1>&2
        return 1
    fi
    # Split at ":"
    local IFS=":"
    set ${CI_JOB_NAME}
    # 5-6 parts only
    if [ $# -lt 5 ]; then
        echo "${errorMsg}" 1>&2
        return 1
    fi
    COMPILER="$2"
    COMPILER_VERSION="$3"
    GPI_VERSION="$4"
    BUILD_TYPE="$5"
    TF_ENV="${6:-tf2.0}"
    if [ "$COMPILER" == "" ] || [ "$COMPILER_VERSION" == "" ] \
    || [ "$GPI_VERSION" == "" ] || [ "$TF_ENV" == "" ]; then
        echo "${errorMsg}" 1>&2
        return 1
    fi
    
    COMPILER_MODULE=`module avail -t 2>&1  | grep ${COMPILER}/${COMPILER_VERSION} | head -n 1`
    GPI_MODULE=`module avail -t 2>&1  | grep ${GPI_VERSION} | head -n 1`
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

parseJobProperties
