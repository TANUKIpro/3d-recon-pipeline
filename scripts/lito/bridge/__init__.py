# Bridge scripts intentionally NOT imported from clip2mesh code paths.
# These run inside /opt/ml-lito/.venv (torch 2.9 + numpy 2.x) and would
# fail to import in the main clip2mesh image (torch 2.5 + numpy 1.x).
