from setuptools import setup

setup(
    name='vaetki',
    version='1.2.0',
    packages=['vaetki'],
    entry_points={
        'vllm.general_plugins': [
            "vaetki_model = vaetki:register",
        ],
    },
)