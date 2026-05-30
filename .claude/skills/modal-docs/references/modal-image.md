# modal.Image

```python
class Image(modal.object.Object)
```

Base class for container images to run functions in.

Do not construct this class directly; instead use one of its static factory methods,
such as `modal.Image.debian_slim`, `modal.Image.from_registry`, or `modal.Image.micromamba`.

## add\_local\_file

```python
def add_local_file(self, local_path: Union[str, Path], remote_path: str, *, copy: bool = False) -> "_Image":
```

Adds a local file to the image at `remote_path` within the container

By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
which speeds up deployment.

Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
[`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
build steps whenever the included files change, but it is required if you want to run additional
build steps after this one.

*Added in v0.66.40*: This method replaces the deprecated `modal.Image.copy_local_file` method.

## add\_local\_dir

```python
def add_local_dir(
    self,
    local_path: Union[str, Path],
    remote_path: str,
    *,
    copy: bool = False,
    # Predicate filter function for file exclusion, which should accept a filepath and return `True` for exclusion.
    # Defaults to excluding no files. If a Sequence is provided, it will be converted to a FilePatternMatcher.
    # Which follows dockerignore syntax.
    ignore: Union[Sequence[str], Callable[[Path], bool]] = [],
) -> "_Image":
```

Adds a local directory's content to the image at `remote_path` within the container

By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
which speeds up deployment.

Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
[`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
build steps whenever the included files change, but it is required if you want to run additional
build steps after this one.

**Usage:**

```python
from modal import FilePatternMatcher

image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=["*.venv"],
)

image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=lambda p: p.is_relative_to(".venv"),
)

image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=FilePatternMatcher("**/*.txt"),
)

# When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=~FilePatternMatcher("**/*.py"),
)

# You can also read ignore patterns from a file.
image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=FilePatternMatcher.from_file("/path/to/ignorefile"),
)
```

*Added in v0.66.40*: This method replaces the deprecated `modal.Image.copy_local_dir` method.

## add\_local\_python\_source

```python
def add_local_python_source(
    self, *modules: str, copy: bool = False, ignore: Union[Sequence[str], Callable[[Path], bool]] = NON_PYTHON_FILES
) -> "_Image":
```

Adds locally available Python packages/modules to containers

Adds all files from the specified Python package or module to containers running the Image.

Packages are added to the `/root` directory of containers, which is on the `PYTHONPATH`
of any executed Modal Functions, enabling import of the module by that name.

By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
which speeds up deployment.

Set `copy=True` to copy the files into an Image layer at build time instead. This can slow down iteration since
it requires a rebuild of the Image and any subsequent build steps whenever the included files change, but it is
required if you want to run additional build steps after this one.

**Note:** This excludes all dot-prefixed subdirectories or files and all `.pyc`/`__pycache__` files.
To add full directories with finer control, use `.add_local_dir()` instead and specify `/root` as
the destination directory.

By default only includes `.py`-files in the source modules. Set the `ignore` argument to a list of patterns
or a callable to override this behavior, e.g.:

```py
# includes everything except data.json
modal.Image.debian_slim().add_local_python_source("mymodule", ignore=["data.json"])

# exclude large files
modal.Image.debian_slim().add_local_python_source(
    "mymodule",
    ignore=lambda p: p.stat().st_size > 1e9
)
```

*Added in v0.67.28*: This method replaces the deprecated `modal.Mount.from_local_python_packages` pattern.

## from\_id

```python
@deprecate_aio_usage((2025, 11, 14), "Image.from_id")
@classmethod
def from_id(cls, image_id: str, client: Optional["modal.client.Client"] = None) -> typing_extensions.Self:
```

Construct an Image from an id and look up the Image result.

The ID of an Image object can be accessed using `.object_id`.

## build

```python
def build(self, app: "modal.app._App") -> "_Image":
```

Eagerly build an image.

If your image was previously built, then this method will not rebuild your image
and your cached image is returned.

**Examples**

```python
image = modal.Image.debian_slim().uv_pip_install("scipy", "numpy")

app = modal.App.lookup("build-image", create_if_missing=True)
with modal.enable_output():  # To see logs in your local terminal
    image.build(app)

# Save the image id
my_image_id = image.object_id

# Reference the image with the id or uses it another context.
built_image = modal.Image.from_id(my_image_id)
```

Alternatively, you can pre-build a image and use it in a sandbox.

```python notest
app = modal.App.lookup("sandbox-example", create_if_missing=True)

with modal.enable_output():
    image = modal.Image.debian_slim().uv_pip_install("scipy")
    image.build(app)

sb = modal.Sandbox.create("python", "-c", "import scipy; print(scipy)", app=app, image=image)
print(sb.stdout.read())
sb.terminate()
```

**Note**

For defining Modal functions, images are built automatically when deploying or running an App.
You do not need to built the image explicitly:

```python notest
app = modal.App()
image = modal.Image.debian_slim()

# No need to explicitly build the image for defining a function.
@app.function(image=image)
def f():
    ...
```

## pip\_install

```python
def pip_install(
    self,
    *packages: Union[str, list[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Install a list of Python packages using pip.

**Examples**

Simple installation:

```python
image = modal.Image.debian_slim().pip_install("click", "httpx~=0.23.3")
```

More complex installation:

```python
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "transformers==4.40.2",
    )
    .pip_install(
        "flash-attn==2.5.8", extra_options="--no-build-isolation"
    )
)
```

## pip\_install\_private\_repos

```python
def pip_install_private_repos(
    self,
    *repositories: str,
    git_user: str,
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
) -> "_Image":
```

Install a list of Python packages from private git repositories using pip.

This method currently supports Github and Gitlab only.

* **Github:** Provide a `modal.Secret` that contains a `GITHUB_TOKEN` key-value pair
* **Gitlab:** Provide a `modal.Secret` that contains a `GITLAB_TOKEN` key-value pair

These API tokens should have permissions to read the list of private repositories provided as arguments.

We recommend using Github's ['fine-grained' access tokens](https://github.blog/2022-10-18-introducing-fine-grained-personal-access-tokens-for-github/).
These tokens are repo-scoped, and avoid granting read permission across all of a user's private repos.

**Example**

```python
image = (
    modal.Image
    .debian_slim()
    .pip_install_private_repos(
        "github.com/ecorp/private-one@1.0.0",
        "github.com/ecorp/private-two@main"
        "github.com/ecorp/private-three@d4776502"
        # install from 'inner' directory on default branch.
        "github.com/ecorp/private-four#subdirectory=inner",
        git_user="erikbern",
        secrets=[modal.Secret.from_name("github-read-private")],
    )
)
```

## pip\_install\_from\_requirements

```python
def pip_install_from_requirements(
    self,
    requirements_txt: str,  # Path to a requirements.txt file.
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    *,
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    env: Optional[dict[str, Optional[str]]] = None,
    secrets: Optional[Collection[_Secret]] = None,
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Install a list of Python packages from a local `requirements.txt` file.

## pip\_install\_from\_pyproject

```python
def pip_install_from_pyproject(
    self,
    pyproject_toml: str,
    optional_dependencies: list[str] = [],
    *,
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Install dependencies specified by a local `pyproject.toml` file.

`optional_dependencies` is a list of the keys of the
optional-dependencies section(s) of the `pyproject.toml` file
(e.g. test, doc, experiment, etc). When provided,
all of the packages in each listed section are installed as well.

## uv\_pip\_install

```python
def uv_pip_install(
    self,
    *packages: Union[str, list[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    requirements: Optional[list[str]] = None,  # Passes -r (--requirements) to uv pip install
    find_links: Optional[str] = None,  # Passes -f (--find-links) to uv pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to uv pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to uv pip install
    pre: bool = False,  # Allow pre-releases using uv pip install --prerelease allow
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    uv_version: Optional[str] = None,  # uv version to use
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Install a list of Python packages using uv pip install.

**Examples**

Simple installation:

```python
image = modal.Image.debian_slim().uv_pip_install("torch==2.7.1", "numpy")
```

This method assumes that:

* Python is on the `$PATH` and dependencies are installed with the first Python on the `$PATH`.
* Shell supports backticks for substitution
* `which` command is on the `$PATH`

Added in v1.1.0.

## poetry\_install\_from\_file

```python
def poetry_install_from_file(
    self,
    poetry_pyproject_toml: str,
    poetry_lockfile: Optional[str] = None,  # Path to lockfile. If not provided, uses poetry.lock in same directory.
    *,
    ignore_lockfile: bool = False,  # If set to True, do not use poetry.lock, even when present
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    # Selected optional dependency groups to install (See https://python-poetry.org/docs/cli/#install)
    with_: list[str] = [],
    # Selected optional dependency groups to exclude (See https://python-poetry.org/docs/cli/#install)
    without: list[str] = [],
    only: list[str] = [],  # Only install dependency groups specifed in this list.
    poetry_version: Optional[str] = "latest",  # Version of poetry to install, or None to skip installation
    # If set to True, use old installer. See https://github.com/python-poetry/poetry/issues/3336
    old_installer: bool = False,
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Install poetry *dependencies* specified by a local `pyproject.toml` file.

If not provided as argument the path to the lockfile is inferred. However, the
file has to exist, unless `ignore_lockfile` is set to `True`.

Note that the root project of the poetry project is not installed, only the dependencies.
For including local python source files see `add_local_python_source`

Poetry will be installed to the Image (using pip) unless `poetry_version` is set to None.
Note that the interpretation of `poetry_version="latest"` depends on the Modal Image Builder
version, with versions 2024.10 and earlier limiting poetry to 1.x.

## uv\_sync

```python
def uv_sync(
    self,
    uv_project_dir: str = "./",  # Path to local uv managed project
    *,
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    groups: Optional[list[str]] = None,  # Dependency group to install using `uv sync --group`
    extras: Optional[list[str]] = None,  # Optional dependencies to install using `uv sync --extra`
    frozen: bool = True,  # If True, then we run `uv sync --frozen` when a uv.lock file is present
    extra_options: str = "",  # Extra options to pass to `uv sync`
    uv_version: Optional[str] = None,  # uv version to use
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Creates a virtual environment with the dependencies in a uv managed project with `uv sync`.

**Examples**

```python
image = modal.Image.debian_slim().uv_sync()
```

The `pyproject.toml` and `uv.lock` in `uv_project_dir` are automatically added to the build context. The
`uv_project_dir` is relative to the current working directory of where `modal` is called.

NOTE: This does *not* install the project itself into the environment (this is equivalent to the
`--no-install-project` flag in the `uv sync` command) and you would be expected to add any local python source
files using `Image.add_local_python_source` or similar methods after this call.

This ensures that updates to your project code wouldn't require reinstalling third-party dependencies
after every change.

uv workspaces are currently not supported.

Added in v1.1.0.

## dockerfile\_commands

```python
def dockerfile_commands(
    self,
    *dockerfile_commands: Union[str, list[str]],
    context_files: dict[str, str] = {},
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
    context_dir: Optional[Union[Path, str]] = None,  # Context for relative COPY commands
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    ignore: Union[Sequence[str], Callable[[Path], bool]] = AUTO_DOCKERIGNORE,
    build_args: dict[str, str] = {},  # Dockerfile variables to set
) -> "_Image":
```

Extend an image with arbitrary Dockerfile-like commands.

**Usage:**

```python
from modal import FilePatternMatcher

# By default a .dockerignore file is used if present in the current working directory
image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
)

image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=["*.venv"],
)

image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=lambda p: p.is_relative_to(".venv"),
)

image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=FilePatternMatcher("**/*.txt"),
)

# When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=~FilePatternMatcher("**/*.py"),
)

# You can also read ignore patterns from a file.
image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=FilePatternMatcher.from_file("/path/to/dockerignore"),
)
```

## entrypoint

```python
def entrypoint(
    self,
    entrypoint_commands: list[str],
) -> "_Image":
```

Set the ENTRYPOINT for the image.

## shell

```python
def shell(
    self,
    shell_commands: list[str],
) -> "_Image":
```

Overwrite default shell for the image.

## run\_commands

```python
def run_commands(
    self,
    *commands: Union[str, list[str]],
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    volumes: Optional[dict[Union[str, PurePosixPath], _Volume]] = None,  # Volume mount paths
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
) -> "_Image":
```

Extend an image with a list of shell commands to run.

## micromamba

```python
@staticmethod
def micromamba(
    python_version: Optional[str] = None,
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
) -> "_Image":
```

A Micromamba base image. Micromamba allows for fast building of small Conda-based containers.

## micromamba\_install

```python
def micromamba_install(
    self,
    # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    *packages: Union[str, list[str]],
    # A local path to a file containing package specifications
    spec_file: Optional[str] = None,
    # A list of Conda channels, eg. ["conda-forge", "nvidia"].
    channels: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Install a list of additional packages using micromamba.

## from\_registry

```python
@staticmethod
def from_registry(
    tag: str,
    secret: Optional[_Secret] = None,
    *,
    setup_dockerfile_commands: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    add_python: Optional[str] = None,
    **kwargs,
) -> "_Image":
```

Build a Modal Image from a public or private image registry, such as Docker Hub.

The image must be built for the `linux/amd64` platform.

If your image does not come with Python installed, you can use the `add_python` parameter
to specify a version of Python to add to the image. Otherwise, the image is expected to
have Python on PATH as `python`, along with `pip`.

You may also use `setup_dockerfile_commands` to run Dockerfile commands before the
remaining commands run. This might be useful if you want a custom Python installation or to
set a `SHELL`. Prefer `run_commands()` when possible though.

To authenticate against a private registry with static credentials, you must set the `secret` parameter to
a `modal.Secret` containing a username (`REGISTRY_USERNAME`) and
an access token or password (`REGISTRY_PASSWORD`).

To authenticate against private registries with credentials from a cloud provider,
use `Image.from_gcp_artifact_registry()` or `Image.from_aws_ecr()`.

**Examples**

```python
modal.Image.from_registry("python:3.11-slim-bookworm")
modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
```

## from\_gcp\_artifact\_registry

```python
@staticmethod
def from_gcp_artifact_registry(
    tag: str,
    secret: Optional[_Secret] = None,
    *,
    setup_dockerfile_commands: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    add_python: Optional[str] = None,
    **kwargs,
) -> "_Image":
```

Build a Modal image from a private image in Google Cloud Platform (GCP) Artifact Registry.

You will need to pass a `modal.Secret` containing [your GCP service account key data](https://cloud.google.com/iam/docs/keys-create-delete#creating)
as `SERVICE_ACCOUNT_JSON`. This can be done from the [Secrets](https://modal.com/secrets) page.
Your service account should be granted a specific role depending on the GCP registry used:

* For Artifact Registry images (`pkg.dev` domains) use
  the ["Artifact Registry Reader"](https://cloud.google.com/artifact-registry/docs/access-control#roles) role
* For Container Registry images (`gcr.io` domains) use
  the ["Storage Object Viewer"](https://cloud.google.com/artifact-registry/docs/transition/setup-gcr-repo) role

**Note:** This method does not use `GOOGLE_APPLICATION_CREDENTIALS` as that
variable accepts a path to a JSON file, not the actual JSON string.

See `Image.from_registry()` for information about the other parameters.

**Example**

```python
modal.Image.from_gcp_artifact_registry(
    "us-east1-docker.pkg.dev/my-project-1234/my-repo/my-image:my-version",
    secret=modal.Secret.from_name(
        "my-gcp-secret",
        required_keys=["SERVICE_ACCOUNT_JSON"],
    ),
    add_python="3.11",
)
```

## from\_aws\_ecr

```python
@staticmethod
def from_aws_ecr(
    tag: str,
    secret: Optional[_Secret] = None,
    *,
    setup_dockerfile_commands: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    add_python: Optional[str] = None,
    **kwargs,
) -> "_Image":
```

Build a Modal image from a private image in AWS Elastic Container Registry (ECR).

You will need to pass a `modal.Secret` containing either IAM user credentials or OIDC
configuration to access the target ECR registry.

For IAM user authentication, set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION`.

For OIDC authentication, set `AWS_ROLE_ARN` and `AWS_REGION`.

IAM configuration details can be found in the AWS documentation for
["Private repository policies"](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policies.html).

For more details on using an AWS role to access ECR, see the [OIDC integration guide](https://modal.com/docs/guide/oidc-integration).

See `Image.from_registry()` for information about the other parameters.

**Example**

```python
modal.Image.from_aws_ecr(
    "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:my-version",
    secret=modal.Secret.from_name(
        "aws",
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    ),
    add_python="3.11",
)
```

## from\_dockerfile

```python
@staticmethod
def from_dockerfile(
    path: Union[str, Path],  # Filepath to Dockerfile.
    *,
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    context_dir: Optional[Union[Path, str]] = None,  # Context for relative COPY commands
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
    add_python: Optional[str] = None,  # Version of Python to include when base Image does not have one
    build_args: dict[str, str] = {},  # Dockerfile variables to set
    ignore: Union[Sequence[str], Callable[[Path], bool]] = AUTO_DOCKERIGNORE,
) -> "_Image":
```

Build a Modal image from a local Dockerfile.

If your Dockerfile does not have Python installed, you can use the `add_python` parameter
to specify a version of Python to add to the image.

**Usage:**

```python
from modal import FilePatternMatcher

# By default a .dockerignore file is used if present in the current working directory
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
)

image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=["*.venv"],
)

image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=lambda p: p.is_relative_to(".venv"),
)

image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=FilePatternMatcher("**/*.txt"),
)

# When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=~FilePatternMatcher("**/*.py"),
)

# You can also read ignore patterns from a file.
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=FilePatternMatcher.from_file("/path/to/dockerignore"),
)
```

## from\_scratch

```python
@staticmethod
def from_scratch(force_build: bool = False) -> "_Image":
```

Create an empty Image, equivalent to `FROM scratch` in Docker.

The resulting Image has no operating system, shell, or package manager. It is
primarily useful as a lightweight filesystem to mount into a Sandbox via
`Sandbox.mount_image`.

Note that since this Image doesn't contain Python or other standard OS utilities,
higher-level Image build steps like `pip_install` cannot be chained onto it. It also
cannot be used for `modal.Function` execution, which requires a Python interpreter.

**Example**

```python notest
image = modal.Image.from_scratch().add_local_file(local_path, "/bin/my_binary", copy=True)
```

## debian\_slim

```python
@staticmethod
def debian_slim(python_version: Optional[str] = None, force_build: bool = False) -> "_Image":
```

Default image, based on the official `python` Docker images.

## apt\_install

```python
def apt_install(
    self,
    *packages: Union[str, list[str]],  # A list of packages, e.g. ["ssh", "libpq-dev"]
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str] = None,  # GPU type to attach to the builder container
) -> "_Image":
```

Install a list of Debian packages using `apt`.

**Example**

```python
image = modal.Image.debian_slim().apt_install("git")
```

## run\_function

```python
def run_function(
    self,
    raw_f: Callable[..., Any],
    *,
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    volumes: dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]] = {},  # Volume mount paths
    network_file_systems: dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},  # NFS mount paths
    gpu: Optional[str | list[str]] = None,  # GPU type (or list of types) to attach to the builder container
    cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
    memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
    timeout: int = 60 * 60,  # Maximum execution time of the function in seconds.
    cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    args: Sequence[Any] = (),  # Positional arguments to the function.
    kwargs: dict[str, Any] = {},  # Keyword arguments to the function.
    include_source: bool = True,  # Whether the builder container should have the Function's source added
) -> "_Image":
```

Run user-defined function `raw_f` as an image build step.

The function runs like an ordinary Modal Function, accepting a resource configuration and integrating
with Modal features like Secrets and Volumes. Unlike ordinary Modal Functions, any changes to the
filesystem state will be captured on container exit and saved as a new Image.

**Note**

Only the source code of `raw_f`, the contents of `**kwargs`, and any referenced *global* variables
are used to determine whether the image has changed and needs to be rebuilt.
If this function references other functions or variables, the image will not be rebuilt if you
make changes to them. You can force a rebuild by changing the function's source code itself.

**Example**

```python notest

def my_build_function():
    open("model.pt", "w").write("parameters!")

image = (
    modal.Image
        .debian_slim()
        .pip_install("torch")
        .run_function(my_build_function, secrets=[...], mounts=[...])
)
```

## env

```python
def env(self, vars: dict[str, str]) -> "_Image":
```

Sets the environment variables in an Image.

**Example**

```python
image = (
    modal.Image.debian_slim()
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```

## workdir

```python
def workdir(self, path: Union[str, PurePosixPath]) -> "_Image":
```

Set the working directory for subsequent image build steps and function execution.

**Example**

```python
image = (
    modal.Image.debian_slim()
    .run_commands("git clone https://xyz app")
    .workdir("/app")
    .run_commands("yarn install")
)
```

## cmd

```python
def cmd(self, cmd: list[str]) -> "_Image":
```

Set the default command (`CMD`) to run when a container is started.

Used with `modal.Sandbox`. Has no effect on `modal.Function`.

**Example**

```python
image = (
    modal.Image.debian_slim().cmd(["python", "app.py"])
)
```

## pipe

```python
def pipe(
    self,
    func: Callable[Concatenate["_Image", P], "_Image"],
    *args: P.args,
    **kwargs: P.kwargs,
) -> "_Image":
```

Apply a local function to expand the Image recipe.

This method can be useful for defining reusable Image build
recipes that compose well with the fluent Image builder interface.

**Example**

```python
def workspace_setup(image: modal.Image, repo: str) -> modal.Image:
    return image.run_commands(f"git clone {repo}").uv_pip_install(".")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pipe(workspace_setup, "https://github.com/example/repo.git")
)
```

## imports

```python
@contextlib.contextmanager
def imports(self):
```

Used to import packages in global scope that are only available when running remotely.
By using this context manager you can avoid an `ImportError` due to not having certain
packages installed locally.

**Usage:**

```python notest
with image.imports():
    import torch
```
