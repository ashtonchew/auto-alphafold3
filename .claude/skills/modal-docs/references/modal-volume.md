# modal.Volume

```python
class Volume(modal.object.Object)
```

A writeable volume that can be used to share files between one or more Modal functions.

The contents of a volume is exposed as a filesystem. You can use it to share data between different functions, or
to persist durable state across several instances of the same function.

Unlike a networked filesystem, you need to explicitly reload the volume to see changes made since it was mounted.
Similarly, you need to explicitly commit any changes you make to the volume for the changes to become visible
outside the current container.

Concurrent modification is supported, but concurrent modifications of the same files should be avoided! Last write
wins in case of concurrent modification of the same file - any data the last writer didn't have when committing
changes will be lost!

As a result, volumes are typically not a good fit for use cases where you need to make concurrent modifications to
the same file (nor is distributed file locking supported).

Volumes can only be reloaded if there are no open files for the volume - attempting to reload with open files
will result in an error.

**Usage**

```python
import modal

app = modal.App()
volume = modal.Volume.from_name("my-persisted-volume", create_if_missing=True)

@app.function(volumes={"/root/foo": volume})
def f():
    with open("/root/foo/bar.txt", "w") as f:
        f.write("hello")
    volume.commit()  # Persist changes

@app.function(volumes={"/root/foo": volume})
def g():
    volume.reload()  # Fetch latest changes
    with open("/root/foo/bar.txt", "r") as f:
        print(f.read())
```

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.

## objects

```python
objects: VolumeManager
```

Namespace with methods for managing named Volume objects.

### objects.create

```python
def create(
    self,
    name: str,  # Name to use for the new Volume
    *,
    version: Optional[int] = None,  # Experimental: Configure the backend VolumeFS version
    allow_existing: bool = False,  # If True, no-op when the Volume already exists
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> None:
```

Create a new Volume object.

**Examples:**

```python notest
modal.Volume.objects.create("my-volume")
```

Volumes will be created in the active environment, or another one can be specified:

```python notest
modal.Volume.objects.create("my-volume", environment_name="dev")
```

By default, an error will be raised if the Volume already exists, but passing
`allow_existing=True` will make the creation attempt a no-op in this case.

```python notest
modal.Volume.objects.create("my-volume", allow_existing=True)
```

Note that this method does not return a local instance of the Volume. You can use
`modal.Volume.from_name` to perform a lookup after creation.

Added in v1.1.2.

### objects.list

```python
def list(
    self,
    *,
    max_objects: Optional[int] = None,  # Limit requests to this size
    created_before: Optional[Union[datetime, str]] = None,  # Limit based on creation date
    environment_name: str = "",  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> builtins.list["_Volume"]:
```

Return a list of hydrated Volume objects.

**Examples:**

```python
volumes = modal.Volume.objects.list()
print([v.name for v in volumes])
```

Volumes will be retreived from the active environment, or another one can be specified:

```python notest
dev_volumes = modal.Volume.objects.list(environment_name="dev")
```

By default, all named Volumes are returned, newest to oldest. It's also possible to limit the
number of results and to filter by creation date:

```python
volumes = modal.Volume.objects.list(max_objects=10, created_before="2025-01-01")
```

Added in v1.1.2.

### objects.delete

```python
def delete(
    self,
    name: str,  # Name of the Volume to delete
    *,
    allow_missing: bool = False,  # If True, don't raise an error if the Volume doesn't exist
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
):
```

Delete a named Volume.

Warning: This deletes an *entire Volume*, not just a specific file.
Deletion is irreversible and will affect any Apps currently using the Volume.

**Examples:**

```python notest
await modal.Volume.objects.delete("my-volume")
```

Volumes will be deleted from the active environment, or another one can be specified:

```python notest
await modal.Volume.objects.delete("my-volume", environment_name="dev")
```

Added in v1.1.2.

## name

```python
@property
def name(self) -> Optional[str]:
```

## with\_mount\_options

```python
def with_mount_options(
    self,
    *,
    read_only: Optional[bool] = None,
    sub_path: Optional[Union[str, PurePosixPath]] = None,
) -> "_Volume":
```

Configure options used when mounting this Volume.

Note that these options are not properties stored with the Volume itself - they can be individually configured
for each Volume - container association.

read\_only: bool (optional) - set this to True to make the Volume read only from within containers
sub\_path: str | PurePosixPath (optional) - only mount this sub\_path directory from the Volume.
If the directory doesn't exist in the Volume, it will be created when the container starts up

**Mount Volume in read-only mode**

```python
import modal

volume = modal.Volume.from_name("my-volume")

@app.function(volumes={"/mnt": volume.with_mount_options(read_only=True)})
def f():
    return os.mkdir("/mnt/foo")  # not possible!
```

**Mount only part of a Volume using sub\_path**

```python
import modal

volume = modal.Volume.from_name("my-volume")

@app.function(volumes={"/user_data": volume.with_mount_options(sub_path="/users/my_user")})
def f():
    return os.listdir("/user_data")  # lists data from /users/my_user
```

## from\_name

```python
@staticmethod
def from_name(
    name: str,
    *,
    environment_name: Optional[str] = None,
    create_if_missing: bool = False,
    version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
    client: Optional[_Client] = None,
) -> "_Volume":
```

Reference a Volume by name, creating if necessary.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time is is actually used.

```python
vol = modal.Volume.from_name("my-volume", create_if_missing=True)

app = modal.App()

# Volume refers to the same object, even across instances of `app`.
@app.function(volumes={"/data": vol})
def f():
    pass
```

## from\_id

```python
@staticmethod
def from_id(
    volume_id: str,
    client: Optional[_Client] = None,
) -> "_Volume":
```

Construct a Volume from an id and look up the Volume metadata.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time it is actually used.

The ID of a Volume object can be accessed using `.object_id`.

**Example:**

```python notest
@app.function()
def my_worker(volume_id: str):
    vol = modal.Volume.from_id(volume_id)
    for entry in vol.listdir("/"):
        print(entry.path)

with modal.Volume.ephemeral() as vol:
    # Pass the volume ID to a remote function
    my_worker.remote(vol.object_id)
```

## ephemeral

```python
@classmethod
@contextmanager
def ephemeral(
    cls: type["_Volume"],
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
    version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
) -> AsyncGenerator["_Volume", None]:
```

Creates a new ephemeral volume within a context manager:

Usage:

```python
import modal
with modal.Volume.ephemeral() as vol:
    assert vol.listdir("/") == []
```

```python notest
async with modal.Volume.ephemeral() as vol:
    assert await vol.listdir("/") == []
```

## info

```python
@live_method
def info(self) -> VolumeInfo:
```

Return information about the Volume object.

## commit

```python
@live_method
def commit(self):
```

Commit changes to a mounted volume.

If successful, the changes made are now persisted in durable storage and available to other containers accessing
the volume.

## reload

```python
@live_method
def reload(self):
```

Make latest committed state of volume available in the running container.

Any uncommitted changes to the volume, such as new or modified files, may implicitly be committed when
reloading.

Reloading will fail if there are open files for the volume.

## iterdir

```python
@live_method_gen
def iterdir(self, path: str, *, recursive: bool = True) -> Iterator[FileEntry]:
```

Iterate over all files in a directory in the volume.

Passing a directory path lists all files in the directory. For a file path, return only that
file's description. If `recursive` is set to True, list all files and folders under the path
recursively.

## listdir

```python
@live_method
def listdir(self, path: str, *, recursive: bool = False) -> list[FileEntry]:
```

List all files under a path prefix in the modal.Volume.

Passing a directory path lists all files in the directory. For a file path, return only that
file's description. If `recursive` is set to True, list all files and folders under the path
recursively.

## read\_file

```python
@live_method_gen
def read_file(self, path: str) -> AsyncGenerator[bytes, None]:
```

Read a file from the modal.Volume.

Note - this function is primarily intended to be used outside of a Modal App.
For more information on downloading files from a Modal Volume, see
[the guide](https://modal.com/docs/guide/volumes).

**Example:**

```python notest
vol = modal.Volume.from_name("my-modal-volume")
data = b""
for chunk in vol.read_file("1mb.csv"):
    data += chunk
print(len(data))  # == 1024 * 1024
```

## remove\_file

```python
@live_method
def remove_file(self, path: str, recursive: bool = False) -> None:
```

Remove a file or directory from a volume.

## copy\_files

```python
@live_method
def copy_files(self, src_paths: Sequence[str], dst_path: str, recursive: bool = False) -> None:
```

Copy files within the volume from src\_paths to dst\_path.
The semantics of the copy operation follow those of the UNIX cp command.

The `src_paths` parameter is a list. If you want to copy a single file, you should pass a list with a
single element.

`src_paths` and `dst_path` should refer to the desired location *inside* the volume. You do not need to prepend
the volume mount path.

**Usage**

```python notest
vol = modal.Volume.from_name("my-modal-volume")

vol.copy_files(["bar/example.txt"], "bar2")  # Copy files to another directory
vol.copy_files(["bar/example.txt"], "bar/example2.txt")  # Rename a file by copying
```

Note that if the volume is already mounted on the Modal function, you should use normal filesystem operations
like `os.rename()` and then `commit()` the volume. The `copy_files()` method is useful when you don't have
the volume mounted as a filesystem, e.g. when running a script on your local computer.

## batch\_upload

```python
@live_method_contextmanager
@contextmanager
def batch_upload(self, force: bool = False) -> AsyncGenerator["_AbstractVolumeUploadContextManager", None]:
```

Initiate a batched upload to a volume.

To allow overwriting existing files, set `force` to `True` (you cannot overwrite existing directories with
uploaded files regardless).

**Example:**

```python notest
vol = modal.Volume.from_name("my-modal-volume")

with vol.batch_upload() as batch:
    batch.put_file("local-path.txt", "/remote-path.txt")
    batch.put_directory("/local/directory/", "/remote/directory")
    batch.put_file(io.BytesIO(b"some data"), "/foobar")
```

## rename

```python
@staticmethod
def rename(
    old_name: str,
    new_name: str,
    *,
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
):
```
