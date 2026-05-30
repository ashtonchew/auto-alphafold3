# modal.App

```python
class App(object)
```

A Modal App is a group of functions and classes that are deployed together.

The app serves at least three purposes:

* A unit of deployment for functions and classes.
* Syncing of identities of (primarily) functions and classes across processes
  (your local Python interpreter and every Modal container active in your application).
* Manage log collection for everything that happens inside your code.

**Registering functions with an app**

The most common way to explicitly register an Object with an app is through the
`@app.function()` decorator. It both registers the annotated function itself and
other passed objects, like schedules and secrets, with the app:

```python
import modal

app = modal.App()

@app.function(
    secrets=[modal.Secret.from_name("some_secret")],
    schedule=modal.Period(days=1),
)
def foo():
    pass
```

In this example, the secret and schedule are registered with the app.

```python
def __init__(
    self,
    name: Optional[str] = None,
    *,
    tags: Optional[dict[str, str]] = None,  # Additional metadata to set on the App
    image: Optional[_Image] = None,  # Default Image for the App (otherwise default to `modal.Image.debian_slim()`)
    secrets: Sequence[_Secret] = [],  # Secrets to add for all Functions in the App
    volumes: dict[Union[str, PurePosixPath], _Volume] = {},  # Volume mounts to use for all Functions
    include_source: bool = True,  # Default configuration for adding Function source file(s) to the Modal container
) -> None:
```

Construct a new app, optionally with default image, mounts, secrets, or volumes.

```python notest
image = modal.Image.debian_slim().pip_install(...)
secret = modal.Secret.from_name("my-secret")
volume = modal.Volume.from_name("my-data")
app = modal.App(image=image, secrets=[secret], volumes={"/mnt/data": volume})
```

## name

```python
@property
def name(self) -> Optional[str]:
```

The user-provided name of the App.

## app\_id

```python
@property
def app_id(self) -> Optional[str]:
```

Return the app\_id of a running or stopped app.

## description

```python
@property
def description(self) -> Optional[str]:
```

The App's `name`, if available, or a fallback descriptive identifier.

## lookup

```python
@staticmethod
def lookup(
    name: str,
    *,
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
    create_if_missing: bool = False,
) -> "_App":
```

Look up an App with a given name, creating a new App if necessary.

Note that Apps created through this method will be in a deployed state,
but they will not have any associated Functions or Classes. This method
is mainly useful for creating an App to associate with a Sandbox:

```python
app = modal.App.lookup("my-app", create_if_missing=True)
modal.Sandbox.create("echo", "hi", app=app)
```

## get\_dashboard\_url

```python
def get_dashboard_url(self) -> str:
```

## run

```python
@contextmanager
def run(
    self,
    *,
    name: Optional[str] = None,
    client: Optional[_Client] = None,
    detach: bool = False,
    interactive: bool = False,
    environment_name: Optional[str] = None,
) -> AsyncGenerator["_App", None]:
```

Context manager that runs an ephemeral app on Modal.

Use this as the main entry point for your Modal application. All calls
to Modal Functions should be made within the scope of this context
manager, and they will correspond to the current App.

**Example**

```python notest
with app.run():
    some_modal_function.remote()
```

To enable output printing (i.e., to see App logs), use `modal.enable_output()`:

```python notest
with modal.enable_output():
    with app.run():
        some_modal_function.remote()
```

Note that you should not invoke this in global scope of a file where you have
Modal Functions or Classes defined, since that would run the block when the Function
or Cls is imported in your containers as well. If you want to run it as your entrypoint,
consider protecting it:

```python
if __name__ == "__main__":
    with app.run():
        some_modal_function.remote()
```

You can then run your script with:

```shell
python app_module.py
```

## deploy

```python
def deploy(
    self,
    *,
    name: Optional[str] = None,  # Name for the deployment, overriding any set on the App
    environment_name: Optional[str] = None,  # Environment to deploy the App in
    tag: str = "",  # Optional metadata that is specific to this deployment
    client: Optional[_Client] = None,  # Alternate client to use for communication with the server
    # Deployment strategy:
    # - 'rolling' (default): Traffic shifts to the new deployment by starting
    #   new containers gradually. Old containers will continue to process inputs while new
    #   containers start up.
    # - 'recreate': As part of the deployment, all running containers are terminated. New inputs
    #   will start new containers.
    strategy: str = "rolling",
) -> typing_extensions.Self:
```

Deploy the App so that it is available persistently.

Deployed Apps will be available for lookup or web-based invocations until they are stopped.
Unlike with `App.run`, this method will return as soon as the deployment completes.

This method is a programmatic alternative to the `modal deploy` CLI command.

Examples:

```python notest
app = App("my-app")
app.deploy()
```

To enable output printing (i.e., to see build logs), use `modal.enable_output()`:

```python notest
app = App("my-app")
with modal.enable_output():
    app.deploy()
```

Unlike with `App.run`, Function logs will not stream back to the local client after the
App is deployed.

Note that you should not invoke this method in global scope, as that would redeploy
the App every time the file is imported. If you want to write a programmatic deployment
script, protect this call so that it only runs when the file is executed directly:

```python notest
if __name__ == "__main__":
    with modal.enable_output():
        app.deploy()
```

Then you can deploy your app with:

```shell
python app_module.py
```

## local\_entrypoint

```python
def local_entrypoint(
    self, _warn_parentheses_missing: Any = None, *, name: Optional[str] = None
) -> Callable[[Callable[..., Any]], _LocalEntrypoint]:
```

Decorate a function to be used as a CLI entrypoint for a Modal App.

These functions can be used to define code that runs locally to set up the app,
and act as an entrypoint to start Modal functions from. Note that regular
Modal functions can also be used as CLI entrypoints, but unlike `local_entrypoint`,
those functions are executed remotely directly.

**Example**

```python
@app.local_entrypoint()
def main():
    some_modal_function.remote()
```

You can call the function using `modal run` directly from the CLI:

```shell
modal run app_module.py
```

Note that an explicit [`app.run()`](https://modal.com/docs/reference/modal.App#run) is not needed, as an
[app](https://modal.com/docs/guide/apps) is automatically created for you.

**Multiple Entrypoints**

If you have multiple `local_entrypoint` functions, you can qualify the name of your app and function:

```shell
modal run app_module.py::app.some_other_function
```

**Parsing Arguments**

If your entrypoint function take arguments with primitive types, `modal run` automatically parses them as
CLI options.
For example, the following function can be called with `modal run app_module.py --foo 1 --bar "hello"`:

```python
@app.local_entrypoint()
def main(foo: int, bar: str):
    some_modal_function.call(foo, bar)
```

Currently, `str`, `int`, `float`, `bool`, and `datetime.datetime` are supported.
Use `modal run app_module.py --help` for more information on usage.

## function

```python
@handle_deprecated_parameters
def function(
    self,
    *,
    image: Optional[_Image] = None,  # The image to run as the container for the function
    schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str | list[str]] = None,  # GPU request; either a single GPU type or a list of types
    serialized: bool = False,  # Whether to send the function over using cloudpickle.
    network_file_systems: dict[
        Union[str, PurePosixPath], _NetworkFileSystem
    ] = {},  # Mountpoints for Modal NetworkFileSystems
    volumes: dict[
        Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
    ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
    # Specify, in fractional CPU cores, how many CPU cores to request.
    # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
    # CPU throttling will prevent a container from exceeding its specified limit.
    cpu: Optional[Union[float, tuple[float, float]]] = None,
    # Specify, in MiB, a memory request which is the minimum memory required.
    # Or, pass (request, limit) to additionally specify a hard limit in MiB.
    memory: Optional[Union[int, tuple[int, int]]] = None,
    ephemeral_disk: Optional[int] = None,  # Specify, in MiB, the ephemeral disk size for the Function.
    min_containers: Optional[int] = None,  # Minimum number of containers to keep warm, even when Function is idle.
    max_containers: Optional[int] = None,  # Limit on the number of containers that can be concurrently running.
    buffer_containers: Optional[int] = None,  # Number of additional idle containers to maintain under active load.
    scaledown_window: Optional[int] = None,  # Max time (in seconds) a container can remain idle while scaling down.
    proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
    retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
    timeout: int = 300,  # Maximum execution time for inputs and startup time in seconds.
    startup_timeout: Optional[int] = None,  # Maximum startup time in seconds with higher precedence than `timeout`.
    name: Optional[str] = None,  # Sets the Modal name of the function within the app
    is_generator: Optional[
        bool
    ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
    cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
    routing_region: Optional[str] = None,  # Region to route inputs to the function through.
    nonpreemptible: bool = False,  # Whether to run the function on a nonpreemptible instance.
    enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
    block_network: bool = False,  # Whether to block network access
    restrict_modal_access: bool = False,  # Whether to allow this function access to other Modal resources
    single_use_containers: bool = False,  # When True, containers will shut down after handling a single input
    i6pn: Optional[bool] = None,  # Whether to enable IPv6 container networking within the region.
    # Whether the file or directory containing the Function's source should automatically be included
    # in the container. When unset, falls back to the App-level configuration, or is otherwise True by default.
    include_source: Optional[bool] = None,
    experimental_options: Optional[dict[str, Any]] = None,
    # Parameters below here are experimental. Use with caution!
    _experimental_restrict_output: bool = False,  # Don't use pickle for return values
    # Parameters below here are deprecated. Please update your code as suggested
    max_inputs: Optional[int] = None,  # Replaced with `single_use_containers`
) -> _FunctionDecoratorType:
```

Decorator to register a new Modal Function with this App.

## cls

```python
@typing_extensions.dataclass_transform(field_specifiers=(parameter,), kw_only_default=True)
@handle_deprecated_parameters
def cls(
    self,
    *,
    image: Optional[_Image] = None,  # The image to run as the container for the function
    env: Optional[dict[str, Optional[str]]] = None,  # Environment variables to set in the container
    secrets: Optional[Collection[_Secret]] = None,  # Secrets to inject into the container as environment variables
    gpu: Optional[str | list[str]] = None,  # GPU request; either a single GPU type or a list of types
    serialized: bool = False,  # Whether to send the function over using cloudpickle.
    network_file_systems: dict[
        Union[str, PurePosixPath], _NetworkFileSystem
    ] = {},  # Mountpoints for Modal NetworkFileSystems
    volumes: dict[
        Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
    ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
    # Specify, in fractional CPU cores, how many CPU cores to request.
    # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
    # CPU throttling will prevent a container from exceeding its specified limit.
    cpu: Optional[Union[float, tuple[float, float]]] = None,
    # Specify, in MiB, a memory request which is the minimum memory required.
    # Or, pass (request, limit) to additionally specify a hard limit in MiB.
    memory: Optional[Union[int, tuple[int, int]]] = None,
    ephemeral_disk: Optional[int] = None,  # Specify, in MiB, the ephemeral disk size for the Function.
    min_containers: Optional[int] = None,  # Minimum number of containers to keep warm, even when Function is idle.
    max_containers: Optional[int] = None,  # Limit on the number of containers that can be concurrently running.
    buffer_containers: Optional[int] = None,  # Number of additional idle containers to maintain under active load.
    scaledown_window: Optional[int] = None,  # Max time (in seconds) a container can remain idle while scaling down.
    proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
    retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
    timeout: int = 300,  # Maximum execution time for inputs and startup time in seconds.
    startup_timeout: Optional[int] = None,  # Maximum startup time in seconds with higher precedence than `timeout`.
    cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
    routing_region: Optional[str] = None,  # Region to route inputs to the function through.
    nonpreemptible: bool = False,  # Whether to run the function on a non-preemptible instance.
    enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
    block_network: bool = False,  # Whether to block network access
    restrict_modal_access: bool = False,  # Whether to allow this class access to other Modal resources
    single_use_containers: bool = False,  # When True, containers will shut down after handling a single input
    i6pn: Optional[bool] = None,  # Whether to enable IPv6 container networking within the region.
    include_source: Optional[bool] = None,  # When `False`, don't automatically add the App source to the container.
    experimental_options: Optional[dict[str, Any]] = None,
    # Parameters below here are experimental. Use with caution!
    _experimental_restrict_output: bool = False,  # Don't use pickle for return values
    # Parameters below here are deprecated. Please update your code as suggested
    max_inputs: Optional[int] = None,  # Replaced with `single_use_containers`
) -> Callable[[Union[CLS_T, _PartialFunction]], CLS_T]:
```

Decorator to register a new Modal [Cls](https://modal.com/docs/reference/modal.Cls) with this App.

## include

```python
def include(self, /, other_app: "_App", inherit_tags: bool = True) -> typing_extensions.Self:
```

Include another App's objects in this one.

Useful for splitting up Modal Apps across different self-contained files.

```python
app_a = modal.App("a")
@app_a.function()
def foo():
    ...

app_b = modal.App("b")
@app_b.function()
def bar():
    ...

app_a.include(app_b)

@app_a.local_entrypoint()
def main():
    # use function declared on the included app
    bar.remote()
```

When `inherit_tags=True` any tags set on the other App will be inherited by this App
(with this App's tags taking precedence in the case of conflicts).

## set\_tags

```python
def set_tags(self, tags: Mapping[str, str], *, client: Optional[_Client] = None) -> None:
```

Attach key-value metadata to the App.

Tag metadata can be used to add organization-specific context to the App and can be
included in billing reports and other informational APIs. Tags can also be set in
the App constructor.

Any tags set on the App before calling this method will be removed if they are not
included in the argument (i.e., this method does not have `.update()` semantics).

## get\_tags

```python
def get_tags(self, *, client: Optional[_Client] = None) -> dict[str, str]:
```

Get the tags that are currently attached to the App.
