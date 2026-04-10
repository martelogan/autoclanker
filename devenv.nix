{ pkgs, ... }:
{
  packages = [
    pkgs.git
    pkgs.python311
  ];

  env = {
    UV_CACHE_DIR = ".local/dev/uv-cache";
    MISE_DATA_DIR = ".mise/data";
    MISE_CACHE_DIR = ".mise/cache";
    MISE_CONFIG_DIR = ".mise/config";
    MISE_STATE_DIR = ".mise/state";
  };

  enterShell = ''
    export PATH="$PWD/.local/dev/bin:$PWD/.venv/bin:$PATH"
    echo "autoclanker devenv shell active."
  '';
}
