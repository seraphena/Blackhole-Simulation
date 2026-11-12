# Black Hole Simulator

A real-time black hole simulator written in C++ and OpenGL for Windows. Every pixel is ray-traced on the GPU in a GLSL fragment shader, so light bends around the black hole as it would in Schwarzschild spacetime.

![Screenshot]

## Features
- **Gravitational lensing:** light paths are integrated with RK4 using the Schwarzschild geodesic equation `u'' = -u + 1.5u²` (units where the Schwarzschild radius rs = 1)
- **Accretion disk:** temperature-based coloring, Doppler shift and gravitational redshift, plus animated turbulent gas
- **Black hole shadow:** rays that fall past the event horizon render as black
- **Procedural starfield:** three layers of stars and faint galactic haze, so the lensing is visible
- **Gravity-well grid:** an optional embedding diagram showing the curved spacetime
- **Free camera:** orbit in any direction, over the top or underneath, with smooth movement
- Tone mapping, gamma correction and optional anti-aliasing

## Controls
| Input | Action |
|---|---|
| Move cursor | Rotate the view around the black hole (hover mode) |
| Mouse wheel / W / S | Zoom in and out |
| Z / X | Decrease / increase rotation sensitivity |
| M | Switch between hover mode and drag mode |
| Space | Auto-spin (drag mode) |
| G | Toggle gravity-well grid |
| A | Toggle anti-aliasing |
| R | Reset camera |
| Esc | Quit |

## Requirements
- Windows
- A GPU with OpenGL 2.0+ support
- MSVC or MinGW-w64

## Build and run

**MinGW-w64:**
```bash
g++ -O2 blackhole.cpp -o blackhole.exe -lopengl32 -lgdi32 -luser32
./blackhole.exe
```

**MSVC (Developer Command Prompt):**
```bash
cl /O2 /EHsc blackhole.cpp opengl32.lib user32.lib gdi32.lib
blackhole.exe
```

## How it works
1. For each pixel, a ray is cast from the camera.
2. The ray's path is integrated in its orbital plane using the Schwarzschild light-bending equation.
3. If it crosses the disk plane, the disk is shaded using Doppler and gravitational redshift. If it reaches `r = rs`, it is absorbed. If it escapes, it samples the starfield in its final direction.
4. The result is tone mapped and gamma corrected.

All of this runs in the fragment shader inside `blackhole.cpp`.

## License
MIT