# Yocto/Pathtrace: Tiny Path Tracer

In this homework, you will learn how to build a simple path tracer with enough
features to make it robust for many scenes. In particular, you will learn how to

- write camera with depth of field,
- write a complex material,
- write a naive path tracer,
- write a path tracer with multiple importance sampling.

## Framework

The code uses the library [Yocto/GL](https://github.com/xelatihy/yocto-gl),
that is included in this project in the directory `yocto`.
We suggest to consult the documentation for the library that you can find
at the beginning of the header files. Also, since the library is getting improved
during the duration of the course, se suggest that you star it and watch it
on Github, so that you can notified as improvements are made.

In order to compile the code, you have to install
[Xcode](https://apps.apple.com/it/app/xcode/id497799835?mt=12)
on OsX, [Visual Studio 2019](https://visualstudio.microsoft.com/it/vs/) on Windows,
or a modern version of gcc or clang on Linux,
together with the tools [cmake](www.cmake.org) and [ninja](https://ninja-build.org).
The script `scripts/build.sh` will perform a simple build on OsX.
As discussed in class, we prefer to use
[Visual Studio Code](https://code.visualstudio.com), with
[C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) and
[CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
extensions, that we have configured to use for this course.

You will write your code in the file `yocto_pathtrace.cpp` for functions that
are declared in `yocto_pathtrace.h`. Your renderer is called by `ypathtrace.cpp`
both in offline mode and in interactive mode, the latter with the 
`--interactive` flag.

This repository also contains tests that are executed from the command line
as shown in `run.sh`. The rendered images are saved in the `out/` directory.
The results should match the ones in the directory `check/`.

## Functionality (26 points)

In this homework you will implement the following features:

- **Camera Sampling** in functions `pathtrace_samples()`:
  - implement camera sampling using `sample_disk()` for the lens
  - implement camera ray generation by simulating a thin lens camera
  - follow the slides to understand how to structure the code
- **Naive Path tracing** in function `shade_naive()`:
  - implement a naive path tracer using the product formulation
  - you should handle both delta and non-delta brdfs using `is_delta()`
    and the functions below
  - follow the slides to understand how to structure the code
  - you can use the functions `eval_position()`, `eval_shading_normal()`,
    `eval_material()`
  - follow the slides to understand how to structure the code
- **Brdf sampling** in function `eval_brdfcos()`, `sample_brdscos()`
  and `sample_brdfcos_pdf()`:
  - implement brdf evaluation and sampling in the above functions
  - the brdf is a switch over the following lobes stored in a brdf objects
    - matte lobe with color `color`
    - glossy lobe with color `color`, ior `ior`,
      and roughness `roughness`
    - reflective lobe with color `color`, and roughness `roughness`
    - transparent lobe with color `color`, ior `ior`, and roughness `roughness`
  - you can use all the reflectance functions in Yocto/Shading including
    `eval_<lobe>()`, `sample_<lobe>()`, and `sample_<lobe>_pdf()`
  - `sample_brdfcos()` picks a direction based on one of the lobes
  - `sample_brdfcos_pdf()` is the sum of the PDFs using weights `<lobe>_pdf`
    stored in `brdf`
  - follow the slides to understand how to structure the code
- **Delta handling** in function `eval_delta()`, `sample_delta()` and
    `sample_delta_pdf()`:
  - same as above with the corresponding functions
  - follow the slides to understand how to structure the code
- **Light sampling** in function `sample_lights()` and `sample_lights_pdf()`:
  - implement light sampling for both area lights and environment maps
  - lights and their CDFs are already implemented in `init_lights()`
  - follow the slides to understand how to structure the code
- **Path tracing** in function `shade_pathtrace()`:
  - implement a path tracer in the product formulation that uses MIS for
    illumination of the smooth BRDFs
  - the simplest way here is to get naive path tracing to work,
    then cut&paste that code and finally add light sampling using MIS
  - follow the slides to understand how to structure the code

To help out, we left example code in `shade_eyelight()`. You can also check out
Yocto/Trace that implements a similar path tracer; in this case though pay
attention to the various differences. In our opinion, it is probably easier to
follow the slides than to follow Yocto/Trace.

## Extra Credit (8 points)

Here we put options of things you could try to do.
You do not have to do them all, since points are capped to 8.
Choose then ones you want to do. They are all fun!

- **Refraction** in all BRDFs functions (simple):
  - use the functions in Yocto/Math and Yocto/Shading that directly support refraction
  - support both delta and rough refraction
- **Alias method** for sampling CDFs (simple):
  - when we sample CDFs we do so with a bisection search that is slow even if `O(log n)`
  - instead we could apply the well known alias method described [here](http://www.realtimerendering.com/raytracinggems/rtg2/index.html)
  - implement the alias method to sample lights triangles and environment pixels
- **Bilinear Patch** in intersect and sample (medium):
  - our quads are not planar, and doing them as two triangles is bad
  - instead, we can intersect them as bilinear patches as in [here](https://link.springer.com/chapter/10.1007/978-1-4842-4427-2_8)
  - implement these changes by altering intersect, eval and sample functions
  - reference implementation in [pbrt4](https://github.com/mmp/pbrt-v4)
- **Denoising** (medium):
  - add support for denoising using [Intel Open Image Denoise](https://github.com/OpenImageDenoise/oidn)
  - to do this, you need to save albedo (color) and normals in auxiliary buffers
    in additions to the rendered images; you can either write your own shaders or
    modify all shaders by also returning the needed data
  - compile Intel OIDN on your machine
  - add denoising to `get_image()`
  - submit a comparison between noisy, denoised, and reference images
- **Better BSDFs** (hard):
  - the microfacet model as many issues, mostly around energy conservation
  - they all stem from the main issue that we do not simulate multiple bounces
  - we can do this with the technique described [here](https://shuangz.com/projects/layered-sa18/)
  - reference implementation in [pbrt4](https://github.com/mmp/pbrt-v4)
- **Hair Shading** (hard):
  - add support for shading hairs as in pbrt
  - paper description [here](https://www.pbrt.org/hair.pdf)
  - reference implementation [here](https://github.com/mmp/pbrt-v3/blob/master/src/materials/hair.cpp)
- **Large Scenes** (very simple, only a few points):
  - render the supplied large scenes at very high sampling rate to test your renderer
  - ask the professor for the scenes
- **MYOS**, make your own scene (very simple, only a few points):
  - create additional scenes that you can render from models assembled by you

For handing in, the content of the extra credit have to be described in a PDF
called **readme.pdf**, that needs to be submitted together with code and images.
Just put for each extra credit, the name of the extra credit, a paragraph that
describes what you have implemented, one or more result images, and links to
the resources used in your implementation.

You can produce a PDF in any way you want. One possibility is to write the file
as Markdown and convert it in PDF with VSCode plugins or other tool.
In this manner, you can link directly the images that you hand in.

## Submission

To submit the homework, you need to pack a ZIP file that contains the code
you write and the images it generates, i.e. the ZIP with only the
`yocto_pathtrace/` and `out/` directories.
If you are doing extra credit, include also `apps`, other images, and a
**`readme.pdf`** file that describes the list of implemented extra credit.
The file should be called `<cognome>_<nome>_<numero_di_matricola>.zip`,
i.e. `<lastname>_<firstname>_<studentid>.zip`, and you should exclude
all other directories. Send it on Google Classroom.
