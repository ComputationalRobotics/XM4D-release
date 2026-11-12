# Sourced by the install scripts: conda setup and two helpers.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

new_env() {  # new_env <name> <conda create args...>: create and activate; never touches an existing env
    if conda env list | awk '{print $1}' | grep -qx "$1"; then
        echo "conda env '$1' already exists: remove it first (conda env remove -n $1) or set ENV=<new name>" >&2
        exit 1
    fi
    conda create -n "$1" "${@:2}" -y
    conda activate "$1"
}

clone() {  # clone <url> <dir> <commit>: the tested commit; an existing <dir> is used as it is
    if [ -e "$2" ]; then echo "using the existing $2"; return; fi
    git clone "$1" "$2"
    git -C "$2" checkout -q "$3"
}
