"""Exact orbit split on the J_E side: same algorithm as orbit_split.py, with the pipeline patched into J_E coordinates by je_setup."""
import je_setup
exec(open('orbit_split.py').read())
