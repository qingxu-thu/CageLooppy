import numpy as np

from cagingloop.model_io import compute_vertex_normals, load_obj_mesh


def test_load_obj_mesh_reads_vertices_and_faces(tmp_path):
    obj_path = tmp_path / "tri.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        )
    )

    vertices, faces = load_obj_mesh(obj_path)

    assert vertices.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert faces.tolist() == [[0, 1, 2]]


def test_compute_vertex_normals_for_single_triangle():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=int)

    normals = compute_vertex_normals(vertices, faces)

    assert np.allclose(normals, np.array([[0.0, 0.0, 1.0]] * 3))
