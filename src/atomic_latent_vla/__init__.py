"""CR1 annotation tools and the native-JAX Atomic π0.5 policy."""

# Keep the annotation package importable in a lightweight environment.  The
# policy is loaded only inside the OpenPI/JAX environment.
__all__ = ["AtomicPi05", "AtomicPi05Config", "AtomicTargets"]


def __getattr__(name: str):
    if name in __all__:
        from . import pi05

        return getattr(pi05, name)
    raise AttributeError(name)
