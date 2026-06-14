import pytest
import e3x
import scipy.spatial.transform
import h5py

@pytest.fixture(scope="session", autouse=True)
def setup_temp_e3x_cache(tmp_path_factory):
    """
    Creates a temporary cache that lasts exactly as long as the pytest run.
    """
    # 1. Ask pytest for a session-scoped temporary directory
    temp_dir = tmp_path_factory.mktemp("e3x_cache")
    cache_path = temp_dir / "sph.npz"

    # 2. Point e3x to this path. Do NOT write fake data to it!
    # e3x will see the file doesn't exist, calculate the real L=16 table, and save it here.
    e3x.Config.set_spherical_harmonics_cache(str(cache_path))

    yield  # 3. All tests run here. 

    # 4. Clean up the global e3x state
    e3x.Config.set_spherical_harmonics_cache("")
    # (pytest automatically deletes the temp directory when the session ends)


@pytest.fixture
def mock_reciprocal_h5(tmp_path):
    """ Generates a valid test fixture file containing a known crystal sample layout. """
    h5_path = tmp_path / "mock_finder.h5"
    
    # Generate a deterministic orientation matrix with a known 5.0 degree error offset
    rot_true = scipy.spatial.transform.Rotation.from_euler('xyz', [10.0, 20.0, 30.0], degrees=True)
    rot_error = scipy.spatial.transform.Rotation.from_euler('x', [5.0], degrees=True)
    rot_seed = rot_true * rot_error
    
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("sample/a", data=5.43)
        f.create_dataset("sample/b", data=5.43)
        f.create_dataset("sample/c", data=5.43)
        f.create_dataset("sample/alpha", data=90.0)
        f.create_dataset("sample/beta", data=90.0)
        f.create_dataset("sample/gamma", data=90.0)
        f.create_dataset("sample/space_group", data=b"F m -3 m")
        f.create_dataset("sample/U", data=rot_seed.as_matrix())
        
    return str(h5_path), rot_seed.as_matrix(), rot_true.as_matrix()
