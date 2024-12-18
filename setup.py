import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "1.0.0"
REPO_NAME = "Orange_UC_API"
AUTHOR_USER_NAME = "AmitVermaDce"
SRC_REPO = "uc_api"
AUTHOR_EMAIL = "amit.verma@orange.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="API built for Sentiment Analysis & Summarization using FastAPI",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
