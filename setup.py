from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
	"""Add compiler-specific flags for OpenMP and C99 support."""

	def build_extensions(self):
		compiler = self.compiler.compiler_type

		for ext in self.extensions:
			if compiler == "msvc":
				ext.extra_compile_args.append("/openmp")
			else:
				ext.extra_compile_args.extend(["-O3", "-std=c99", "-fopenmp"])
				ext.extra_link_args.append("-fopenmp")

		super().build_extensions()


def get_numpy_include():
	try:
		import numpy
	except ImportError as exc:
		raise RuntimeError(
			"NumPy is required to build biparallel. "
			"Install it first (for example: pip install numpy)."
		) from exc
	return numpy.get_include()


core_extension = Extension(
	name="biparallel.core.core_functions",
	sources=["biparallel/core/core_functions.c"],
	include_dirs=[get_numpy_include(), "biparallel/core"],
	libraries=["gsl", "gslcblas", "fftw3f_threads", "fftw3f", "m"],
	extra_compile_args=[],
	extra_link_args=[],
)


setup(
	ext_modules=[core_extension],
	cmdclass={"build_ext": BuildExt},
)
