import vtk

with open('mesh_is_watertight.txt', 'r') as output_file:
    data = output_file.readlines()

data =sorted(set(data))

for elem in data:
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(elem)[:-4] + 'vtk')
    reader.Update()

    # Get the output from the reader
    vtk_data = reader.GetOutput()
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(vtk_data)
    geometry_filter.Update()

    # Get the output polydata
    polydata = geometry_filter.GetOutput()

    # Create a PLY writer
    ply_writer = vtk.vtkPLYWriter()
    ply_writer.SetFileName(str(elem)[:-1])

    ply_writer.SetInputData(polydata)

    # Write the PLY file
    ply_writer.Write()
