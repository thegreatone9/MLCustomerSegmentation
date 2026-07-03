import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.1"

REPO_NAME = "MLCustomerSegmentation"
AUTHOR_USER_NAME = ""
SRC_REPO = "customerSegmentation"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="Customer Segmentation ML Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/thegreatone9/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/thegreatone9/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
