// =====================================================================
//  Black hole simulator (GPU / OpenGL, Windows)
//  Units: Schwarzschild radius rs = 1.   Light path in a plane: u'' = -u + 1.5 u^2   (u = 1/r)
//  The ray tracing runs in a fragment shader (GLSL), once per pixel, on the graphics card.
// =====================================================================
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/gl.h>
#ifndef APIENTRY
#define APIENTRY
#endif

// ---------------------------------------------------------------------
// 1. Settings
// ---------------------------------------------------------------------
constexpr int    WIN_W = 1100, WIN_H = 700;     // window size in pixels
constexpr double FOV_DEG  = 60.0;
constexpr double PI       = 3.14159265358979;
constexpr double FUNNEL_RIM = 45.0;             // radius where the gravity-well grid ends

// ---------------------------------------------------------------------
// 2. Small vector type (same idea as before)
// ---------------------------------------------------------------------
struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(double s)      const { return Vec3(x * s, y * s, z * s); }
    double length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const { double l = length(); return Vec3(x / l, y / l, z / l); }
};
double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

// The camera is stored as a position + an "up" direction, so it can rotate freely in ALL directions
// (over the top, underneath, upside down) with no limits.
struct Camera {
    Vec3 pos = Vec3(0, 4.36, 19.52);    // where the camera is (it always looks at the black hole)
    Vec3 up  = Vec3(0, 1, 0);           // which way is up for the camera
};

// rotate vector v around a (unit) axis by an angle (Rodrigues' rotation formula)
Vec3 rotateAround(const Vec3& v, const Vec3& axis, double ang) {
    double c = std::cos(ang), s = std::sin(ang);
    return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (1.0 - c));
}

// forward / right / up directions of the camera
void cameraBasis(const Camera& cam, Vec3& fwd, Vec3& right, Vec3& up) {
    fwd   = (Vec3(0, 0, 0) - cam.pos).normalized();
    right = cross(fwd, cam.up).normalized();
    up    = cross(right, fwd);
}

// dYaw turns around the vertical axis, dPitch tips over the top / underneath (no limit)
void orbitCamera(Camera& cam, double dYaw, double dPitch) {
    Vec3 fwd, right, up;
    cameraBasis(cam, fwd, right, up);
    cam.pos = rotateAround(cam.pos, right, dPitch);
    cam.up  = rotateAround(up,      right, dPitch);
    double sgn = (cam.up.y < 0) ? -1.0 : 1.0;           // keep drag direction natural when upside down
    cam.pos = rotateAround(cam.pos, Vec3(0, 1, 0), dYaw * sgn);
    cam.up  = rotateAround(cam.up,  Vec3(0, 1, 0), dYaw * sgn);
    fwd = (Vec3(0, 0, 0) - cam.pos).normalized();       // clean up rounding errors
    cam.up = (cam.up - fwd * dot(cam.up, fwd)).normalized();
}

void zoomCamera(Camera& cam, double factor) {
    double d = clampd(cam.pos.length() * factor, 5.0, 80.0);
    cam.pos = cam.pos.normalized() * d;
}

// ---------------------------------------------------------------------
// 3. Loading the OpenGL 2.0 functions (Windows only ships OpenGL 1.1 directly)
// ---------------------------------------------------------------------
typedef char GLchar_t;
typedef GLuint (APIENTRY *PFN_CreateShader)(GLenum);
typedef void   (APIENTRY *PFN_ShaderSource)(GLuint, GLsizei, const GLchar_t* const*, const GLint*);
typedef void   (APIENTRY *PFN_CompileShader)(GLuint);
typedef void   (APIENTRY *PFN_GetShaderiv)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY *PFN_GetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar_t*);
typedef GLuint (APIENTRY *PFN_CreateProgram)(void);
typedef void   (APIENTRY *PFN_AttachShader)(GLuint, GLuint);
typedef void   (APIENTRY *PFN_LinkProgram)(GLuint);
typedef void   (APIENTRY *PFN_GetProgramiv)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY *PFN_GetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar_t*);
typedef void   (APIENTRY *PFN_UseProgram)(GLuint);
typedef GLint  (APIENTRY *PFN_GetUniformLocation)(GLuint, const GLchar_t*);
typedef void   (APIENTRY *PFN_Uniform1f)(GLint, GLfloat);
typedef void   (APIENTRY *PFN_Uniform2f)(GLint, GLfloat, GLfloat);
typedef void   (APIENTRY *PFN_Uniform3f)(GLint, GLfloat, GLfloat, GLfloat);
typedef void   (APIENTRY *PFN_Uniform1i)(GLint, GLint);

PFN_CreateShader       p_glCreateShader;
PFN_ShaderSource       p_glShaderSource;
PFN_CompileShader      p_glCompileShader;
PFN_GetShaderiv        p_glGetShaderiv;
PFN_GetShaderInfoLog   p_glGetShaderInfoLog;
PFN_CreateProgram      p_glCreateProgram;
PFN_AttachShader       p_glAttachShader;
PFN_LinkProgram        p_glLinkProgram;
PFN_GetProgramiv       p_glGetProgramiv;
PFN_GetProgramInfoLog  p_glGetProgramInfoLog;
PFN_UseProgram         p_glUseProgram;
PFN_GetUniformLocation p_glGetUniformLocation;
PFN_Uniform1f          p_glUniform1f;
PFN_Uniform2f          p_glUniform2f;
PFN_Uniform3f          p_glUniform3f;
PFN_Uniform1i          p_glUniform1i;

constexpr GLenum GL_FRAGMENT_SHADER_ = 0x8B30, GL_VERTEX_SHADER_ = 0x8B31;
constexpr GLenum GL_COMPILE_STATUS_ = 0x8B81, GL_LINK_STATUS_ = 0x8B82, GL_INFO_LOG_LENGTH_ = 0x8B84;

typedef void* (*GetProcFn)(const char*);

bool loadGL(GetProcFn getProc) {
    #define LOAD(name) p_##name = (decltype(p_##name))getProc(#name); if (!p_##name) { std::cout << "missing OpenGL function: " #name "\n"; return false; }
    LOAD(glCreateShader) LOAD(glShaderSource) LOAD(glCompileShader) LOAD(glGetShaderiv)
    LOAD(glGetShaderInfoLog) LOAD(glCreateProgram) LOAD(glAttachShader) LOAD(glLinkProgram)
    LOAD(glGetProgramiv) LOAD(glGetProgramInfoLog) LOAD(glUseProgram) LOAD(glGetUniformLocation)
    LOAD(glUniform1f) LOAD(glUniform2f) LOAD(glUniform3f) LOAD(glUniform1i)
    #undef LOAD
    return true;
}

// ---------------------------------------------------------------------
// 4. The shaders. This GLSL code runs ON THE GPU, once for every pixel.
// ---------------------------------------------------------------------
const char* VERT_SRC = R"GLSL(
#version 120
void main() { gl_Position = gl_Vertex; }
)GLSL";

const char* FRAG_SRC = R"GLSL(
#version 120
uniform vec2  uRes;
uniform vec3  uCamPos, uRight, uUp, uForward;
uniform float uTanHalf, uTime, uExposure;
uniform int   uAA;
uniform int   uCamAbove;

const float DISK_IN  = 3.0;
const float DISK_OUT = 14.0;
const float H        = 0.02;      // step in angle (radians)
const int   NSTEPS   = 750;

// ---------- noise helpers ----------
float hash31(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hash31(i),                     hash31(i + vec3(1.0, 0.0, 0.0)), f.x),
                  mix(hash31(i + vec3(0.0, 1.0, 0.0)), hash31(i + vec3(1.0, 1.0, 0.0)), f.x), f.y);
    float b = mix(mix(hash31(i + vec3(0.0, 0.0, 1.0)), hash31(i + vec3(1.0, 0.0, 1.0)), f.x),
                  mix(hash31(i + vec3(0.0, 1.0, 1.0)), hash31(i + vec3(1.0, 1.0, 1.0)), f.x), f.y);
    return mix(a, b, f.z);
}
float fbm(vec3 p) {
    float a = 0.5;
    float s = 0.0;
    for (int i = 0; i < 3; i++) {
        s += a * vnoise(p);
        p = p * 2.03 + vec3(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s / 0.875;
}

// ---------- background sky: stars in 3 layers + faint galactic haze ----------
vec3 starLayer(vec3 d, float scale, float density, float seed) {
    vec3 p = d * scale;
    vec3 ip = floor(p);
    vec3 fp = p - ip;
    float h = hash31(ip + seed);
    if (h > density) return vec3(0.0);
    vec3 pos = vec3(hash31(ip + seed + 1.3), hash31(ip + seed + 2.7), hash31(ip + seed + 4.1)) * 0.5 + 0.25;
    float dist = length(fp - pos);
    float bright = exp(-dist * dist * 180.0) * (0.4 + 1.6 * hash31(ip + seed + 9.9));
    float temp = hash31(ip + seed + 5.5);
    vec3 tint = mix(vec3(1.0, 0.75, 0.55), vec3(0.65, 0.8, 1.0), temp);
    return tint * bright;
}
vec3 sky(vec3 d) {
    vec3 c = starLayer(d, 40.0, 0.10, 0.0) * 1.3
           + starLayer(d, 85.0, 0.09, 11.0) * 1.0
           + starLayer(d, 170.0, 0.07, 23.0) * 0.8;
    float band = exp(-d.y * d.y * 9.0);
    float haze = fbm(d * 5.0 + vec3(3.0)) * band;
    c += vec3(0.10, 0.12, 0.22) * haze * 0.07;
    c += vec3(0.004, 0.005, 0.010);
    return c;
}

// ---------- accretion disk ----------
vec3 tempColor(float T) {            // temperature -> color (dark red -> orange -> yellow-white -> blue-white)
    float r = smoothstep(0.0, 0.4, T);
    float g = smoothstep(0.25, 1.1, T);
    float b = smoothstep(0.6, 1.7, T);
    return vec3(r, g, b);
}

// r = radius where the ray hit the disk, pd = unit direction of the hit point (in the disk plane),
// n = direction the light travels toward the camera
vec3 diskShade(float r, vec3 pd, vec3 n, out float alpha) {
    float edgeIn  = smoothstep(DISK_IN, DISK_IN + 0.35, r);
    float edgeOut = 1.0 - smoothstep(DISK_OUT - 5.0, DISK_OUT, r);
    alpha = 0.93 * edgeIn * edgeOut;

    // Doppler shift + gravitational redshift
    vec3  vt   = vec3(-pd.z, 0.0, pd.x);                  // direction the gas orbits
    float beta = min(sqrt(0.5 / (r - 1.0)), 0.8);
    float gam  = 1.0 / sqrt(1.0 - beta * beta);
    float dop  = 1.0 / (gam * (1.0 - beta * dot(vt, n)));
    float g    = dop * sqrt(1.0 - 1.0 / r);

    // temperature falls with radius; observed temperature is shifted by g
    float T = pow(DISK_IN / r, 0.75) * g * 0.95;

    // swirling turbulence (two time-shifted layers cross-fade so it never winds up too tight)
    float ang = atan(pd.z, pd.x);
    float period = 16.0;
    float ph0 = fract(uTime / period);
    float ph1 = fract(uTime / period + 0.5);
    float w0  = 1.0 - abs(2.0 * ph0 - 1.0);
    float om  = 2.4 / (r * sqrt(r));                      // inner gas orbits faster
    float a0  = ang - om * ph0 * period;
    float a1  = ang - om * ph1 * period;
    float n0  = fbm(vec3(cos(a0) * 3.0,       sin(a0) * 3.0,       r * 1.1));
    float n1  = fbm(vec3(cos(a1) * 3.0 + 7.3, sin(a1) * 3.0 + 2.1, r * 1.1));
    float turb = mix(n1, n0, w0);

    float B = pow(T, 3.0) * (0.30 + 1.5 * turb);
    vec3 col = tempColor(T) * B * 2.4;

    return col;
}

// ---------- trace one ray; returns color, and "cover" = how much of the sky it hides ----------
vec3 trace(vec3 dir, out float cover) {
    vec3  P  = uCamPos;
    float r0 = length(P);
    vec3  e1 = P / r0;
    float along = dot(dir, e1);
    vec3  perp  = dir - e1 * along;
    float pl = length(perp);
    float b  = max(r0 * pl, 1e-4);
    vec3  e2 = (pl > 1e-6) ? perp / pl : normalize(cross(e1, vec3(0.3, 0.9, 0.1)));

    float u = 1.0 / r0;
    float v = ((along < 0.0) ? 1.0 : -1.0) * sqrt(max(0.0, 1.0 / (b * b) - u * u + u * u * u));

    float ch = cos(H), sh = sin(H);
    float c = 1.0, s = 0.0;                                // cos and sin of the swept angle phi
    float prevY = e1.y;
    vec3  col   = vec3(0.0);
    float trans = 1.0;
    bool  escaped = false;

    for (int i = 0; i < NSTEPS; i++) {
        if (u >= 1.0)   break;                             // swallowed
        if (u <= 0.001) { escaped = true; break; }         // escaped to infinity

        // one RK4 step of  u' = v,  v' = -u + 1.5 u^2
        vec2 s0 = vec2(u, v);
        vec2 k1 = vec2(s0.y, -s0.x + 1.5 * s0.x * s0.x);
        vec2 t2 = s0 + 0.5 * H * k1;
        vec2 k2 = vec2(t2.y, -t2.x + 1.5 * t2.x * t2.x);
        vec2 t3 = s0 + 0.5 * H * k2;
        vec2 k3 = vec2(t3.y, -t3.x + 1.5 * t3.x * t3.x);
        vec2 t4 = s0 + H * k3;
        vec2 k4 = vec2(t4.y, -t4.x + 1.5 * t4.x * t4.x);
        vec2 s1 = s0 + (H / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        float cPrev = c, sPrev = s;
        float cn = c * ch - s * sh;
        float sn = s * ch + c * sh;
        c = cn; s = sn;

        float y = e1.y * c + e2.y * s;                     // height of the ray over the disk plane

        if (y * prevY < 0.0) {                             // crossed the plane y = 0
            float t  = prevY / (prevY - y);
            float uc = mix(s0.x, s1.x, t);
            float vc = mix(s0.y, s1.y, t);
            float r  = 1.0 / uc;
            if (r > DISK_IN - 0.5 && r < DISK_OUT) {
                float cc = mix(cPrev, c, t);
                float ss = mix(sPrev, s, t);
                vec3 pd   = normalize(e1 * cc + e2 * ss);
                vec3 tang = normalize(-e1 * ss + e2 * cc);
                float rp  = -vc / (uc * uc);               // dr/dphi
                vec3 trav = normalize(rp * pd + r * tang); // direction the traced ray moves
                float alpha;
                vec3 em = diskShade(r, pd, -trav, alpha);
                col   += trans * alpha * em;               // thin, slightly see-through disk
                trans *= (1.0 - alpha);
            }
        }
        prevY = y;
        u = s1.x; v = s1.y;
    }

    if (escaped) {
        float r = 1.0 / u;
        vec3 pd   = e1 * c + e2 * s;
        vec3 tang = -e1 * s + e2 * c;
        float rp  = -v / (u * u);
        vec3 outDir = normalize(rp * pd + r * tang);
        col += trans * sky(outDir);
        cover = 1.0 - trans;
    } else {
        cover = 1.0;                                       // black hole shadow
    }
    return col;
}

void main() {
    float aspect = uRes.x / uRes.y;
    int ns = (uAA >= 2) ? 4 : 1;
    vec3  sum = vec3(0.0);
    float cov = 0.0;
    for (int k = 0; k < 4; k++) {
        if (k >= ns) break;
        vec2 off = (ns == 1) ? vec2(0.0)
                             : (vec2(mod(float(k), 2.0), floor(float(k) / 2.0)) - 0.5) * 0.5;
        vec2 uv = (gl_FragCoord.xy + off) / uRes * 2.0 - 1.0;
        vec3 dir = normalize(uForward + uRight * (uv.x * aspect * uTanHalf) + uUp * (uv.y * uTanHalf));
        float cv;
        sum += trace(dir, cv);
        cov += cv;
    }
    vec3 col = sum / float(ns);
    cov /= float(ns);

    col = vec3(1.0) - exp(-col * uExposure);               // tone mapping (keeps bright parts from clipping)
    col = pow(col, vec3(1.0 / 2.2));                       // gamma
    gl_FragColor = vec4(col, 1.0);

    // The gravity-well grid is drawn after this. Where the disk or hole is, hide the grid behind it.
    gl_FragDepth = (uCamAbove == 1 && cov > 0.5) ? 0.0 : 1.0;
}
)GLSL";

GLuint g_prog = 0;

GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = p_glCreateShader(type);
    p_glShaderSource(sh, 1, &src, NULL);
    p_glCompileShader(sh);
    GLint ok = 0;
    p_glGetShaderiv(sh, GL_COMPILE_STATUS_, &ok);
    if (!ok) {
        GLint len = 0;
        p_glGetShaderiv(sh, GL_INFO_LOG_LENGTH_, &len);
        std::vector<char> log(len + 1, 0);
        p_glGetShaderInfoLog(sh, len, NULL, log.data());
        std::cout << "SHADER COMPILE ERROR:\n" << log.data() << "\n";
        return 0;
    }
    return sh;
}

bool initShaders() {
    GLuint vs = compileShader(GL_VERTEX_SHADER_, VERT_SRC);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER_, FRAG_SRC);
    if (!vs || !fs) return false;
    g_prog = p_glCreateProgram();
    p_glAttachShader(g_prog, vs);
    p_glAttachShader(g_prog, fs);
    p_glLinkProgram(g_prog);
    GLint ok = 0;
    p_glGetProgramiv(g_prog, GL_LINK_STATUS_, &ok);
    if (!ok) {
        char log[2048] = {0};
        p_glGetProgramInfoLog(g_prog, 2047, NULL, log);
        std::cout << "SHADER LINK ERROR:\n" << log << "\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// 5. Drawing one frame: the shader quad + the gravity-well grid
// ---------------------------------------------------------------------
double funnelY(double r) {                 // "embedding diagram" height of the warped sheet
    return 0.6 * (2.0 * std::sqrt(std::max(r - 1.0, 0.0)) - 2.0 * std::sqrt(FUNNEL_RIM - 1.0));
}

void drawFunnel() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.2f);
    const int SEG = 120;

    for (int k = 1; k <= 30; ++k) {                                   // rings
        double r = 1.02 + 0.055 * k * k;
        if (r > FUNNEL_RIM) break;
        float a = (float)(0.75 * (1.0 - r / (FUNNEL_RIM + 5.0)));
        glColor4f(0.55f, 0.65f, 0.95f, a);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < SEG; ++i) {
            double ang = 2.0 * PI * i / SEG;
            double x = r * std::cos(ang), z = r * std::sin(ang);
            glVertex3d(x, funnelY(r), z);
        }
        glEnd();
    }
    const int SPOKES = 36;
    for (int j = 0; j < SPOKES; ++j) {                                // spokes
        double ang = 2.0 * PI * j / SPOKES;
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= 90; ++i) {
            double t = i / 90.0;
            double r = 1.02 + (FUNNEL_RIM - 1.02) * t * t;
            float a = (float)(0.75 * (1.0 - r / (FUNNEL_RIM + 5.0)));
            glColor4f(0.55f, 0.65f, 0.95f, a);
            double x = r * std::cos(ang), z = r * std::sin(ang);
            glVertex3d(x, funnelY(r), z);
        }
        glEnd();
    }
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
}

void drawScene(int w, int h, const Camera& cam, double time, bool showGrid, bool aa) {
    Vec3 camPos = cam.pos;
    Vec3 fwd, right, up;
    cameraBasis(cam, fwd, right, up);
    double tanHalf = std::tan(FOV_DEG * PI / 180.0 / 2.0);

    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ---- pass 1: full-screen quad, the shader does all the black hole work ----
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glDepthMask(GL_TRUE);

    p_glUseProgram(g_prog);
    p_glUniform2f(p_glGetUniformLocation(g_prog, "uRes"), (float)w, (float)h);
    p_glUniform3f(p_glGetUniformLocation(g_prog, "uCamPos"), (float)camPos.x, (float)camPos.y, (float)camPos.z);
    p_glUniform3f(p_glGetUniformLocation(g_prog, "uRight"),  (float)right.x, (float)right.y, (float)right.z);
    p_glUniform3f(p_glGetUniformLocation(g_prog, "uUp"),     (float)up.x,    (float)up.y,    (float)up.z);
    p_glUniform3f(p_glGetUniformLocation(g_prog, "uForward"),(float)fwd.x,   (float)fwd.y,   (float)fwd.z);
    p_glUniform1f(p_glGetUniformLocation(g_prog, "uTanHalf"), (float)tanHalf);
    p_glUniform1f(p_glGetUniformLocation(g_prog, "uTime"), (float)time);
    p_glUniform1f(p_glGetUniformLocation(g_prog, "uExposure"), 1.15f);
    p_glUniform1i(p_glGetUniformLocation(g_prog, "uAA"), aa ? 2 : 1);
    p_glUniform1i(p_glGetUniformLocation(g_prog, "uCamAbove"), camPos.y > 0 ? 1 : 0);

    glBegin(GL_QUADS);
    glVertex3f(-1, -1, 0); glVertex3f(1, -1, 0); glVertex3f(1, 1, 0); glVertex3f(-1, 1, 0);
    glEnd();
    p_glUseProgram(0);

    // ---- pass 2: the gravity-well grid, drawn with a normal 3D camera ----
    if (showGrid) {
        double f = 1.0 / tanHalf, n = 0.1, fa = 400.0, aspect = (double)w / h;
        double proj[16] = {0};
        proj[0] = f / aspect; proj[5] = f;
        proj[10] = (fa + n) / (n - fa); proj[11] = -1.0; proj[14] = 2.0 * fa * n / (n - fa);
        double view[16] = {0};
        view[0] = right.x; view[4] = right.y; view[8]  = right.z;
        view[1] = up.x;    view[5] = up.y;    view[9]  = up.z;
        view[2] = -fwd.x;  view[6] = -fwd.y;  view[10] = -fwd.z;
        view[12] = -dot(right, camPos); view[13] = -dot(up, camPos); view[14] = dot(fwd, camPos);
        view[15] = 1.0;
        glMatrixMode(GL_PROJECTION); glLoadMatrixd(proj);
        glMatrixMode(GL_MODELVIEW);  glLoadMatrixd(view);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        drawFunnel();
        glDepthMask(GL_TRUE);
    }
}

// ---------------------------------------------------------------------
// 6. The window (Windows API + OpenGL) : input and main loop
// ---------------------------------------------------------------------
Camera g_cam;
bool g_running = true, g_showGrid = true, g_autoSpin = false, g_aa = false;
bool g_dragging = false;
int  g_lastX = 0, g_lastY = 0;

// VIEW CONTROL.  Hover mode (default): your cursor position sets the viewing angle, no clicking needed.
//   cursor left/right -> go around the black hole (full 360 degrees across the window)
//   cursor up/down    -> look from underneath ... to from above
// The black hole stays FIXED in space; only the point of view moves.
bool   g_hover = true;
double g_sens  = 1.0;                       // 1.0 = a full 360 degree turn from the left edge to the right edge
double g_yaw = 0.0,  g_pitch = 0.22;        // current (smoothed) viewing angles
double g_tYaw = 0.0, g_tPitch = 0.22;       // angles the cursor is asking for

Vec3 positionFromAngles(double yaw, double pitch, double dist) {
    return Vec3(dist * std::cos(pitch) * std::sin(yaw), dist * std::sin(pitch), dist * std::cos(pitch) * std::cos(yaw));
}

void updateHoverCamera(HWND hwnd, double dt) {
    POINT pt; GetCursorPos(&pt); ScreenToClient(hwnd, &pt);
    RECT rc; GetClientRect(hwnd, &rc);
    int cw = rc.right, ch = rc.bottom;
    if (cw > 8 && ch > 8 && pt.x >= 0 && pt.y >= 0 && pt.x < cw && pt.y < ch) {
        g_tYaw   = -((double)pt.x / cw - 0.5) * 2.0 * PI * g_sens;
        g_tPitch = clampd(0.2 + ((double)pt.y / ch - 0.5) * 2.9 * g_sens, -1.52, 1.52);
    }
    double k = 1.0 - std::exp(-dt * 10.0);                 // ease toward the target (smooth, no jumps)
    g_yaw   += (g_tYaw - g_yaw) * k;
    g_pitch += (g_tPitch - g_pitch) * k;
    double dist = g_cam.pos.length();
    g_cam.pos = positionFromAngles(g_yaw, g_pitch, dist);
    g_cam.up  = Vec3(0, 1, 0);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_LBUTTONDOWN:
        g_dragging = true; SetCapture(hwnd);
        g_lastX = (short)LOWORD(lParam); g_lastY = (short)HIWORD(lParam);
        return 0;
    case WM_LBUTTONUP:
        g_dragging = false; ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE:
        if (g_dragging && !g_hover) {                      // (drag mode only) rotate freely by dragging
            int x = (short)LOWORD(lParam), y = (short)HIWORD(lParam);
            orbitCamera(g_cam, -(x - g_lastX) * 0.006, -(y - g_lastY) * 0.006);
            g_lastX = x; g_lastY = y;
        }
        return 0;
    case WM_MOUSEWHEEL: {
        int delta = (short)HIWORD(wParam);
        zoomCamera(g_cam, delta > 0 ? 0.93 : 1.07);
        return 0;
    }
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) DestroyWindow(hwnd);
        if (wParam == 'G') g_showGrid = !g_showGrid;
        if (wParam == 'A') g_aa = !g_aa;
        if (wParam == VK_SPACE) g_autoSpin = !g_autoSpin;
        if (wParam == 'W') zoomCamera(g_cam, 0.95);
        if (wParam == 'S') zoomCamera(g_cam, 1.05);
        if (wParam == 'Z') g_sens = clampd(g_sens * 0.85, 0.1, 3.0);      // less sensitive
        if (wParam == 'X') g_sens = clampd(g_sens / 0.85, 0.1, 3.0);      // more sensitive
        if (wParam == 'M') {                                              // switch hover mode <-> drag mode
            g_hover = !g_hover;
            if (g_hover) {
                g_yaw = g_tYaw = std::atan2(g_cam.pos.x, g_cam.pos.z);
                g_pitch = g_tPitch = std::asin(clampd(g_cam.pos.y / g_cam.pos.length(), -1.0, 1.0));
            }
        }
        if (wParam == 'R') {                                              // reset everything
            g_cam = Camera(); g_yaw = g_tYaw = 0.0; g_pitch = g_tPitch = 0.22;
        }
        return 0;
    case WM_DESTROY:
        g_running = false; PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

void* winGetProc(const char* name) { return (void*)wglGetProcAddress(name); }

int main() {
    WNDCLASSA wc = {};
    wc.style         = CS_OWNDC;
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = GetModuleHandleA(NULL);
    wc.hCursor       = LoadCursorA(NULL, IDC_ARROW);
    wc.lpszClassName = "BlackHoleGL";
    RegisterClassA(&wc);

    RECT rect = { 0, 0, WIN_W, WIN_H };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowA("BlackHoleGL", "Black hole", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                              80, 40, rect.right - rect.left, rect.bottom - rect.top,
                              NULL, NULL, wc.hInstance, NULL);
    HDC hdc = GetDC(hwnd);

    // ask Windows for an OpenGL-capable pixel format, then create the OpenGL context
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize      = sizeof(pfd);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    int fmt = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, fmt, &pfd);
    HGLRC ctx = wglCreateContext(hdc);
    wglMakeCurrent(hdc, ctx);

    std::cout << "GPU: " << glGetString(GL_RENDERER) << "\nOpenGL: " << glGetString(GL_VERSION) << "\n";
    if (!loadGL(winGetProc) || !initShaders()) {
        std::cout << "\nSetup failed. Copy the message above and send it to me.\n";
        system("pause");
        return 1;
    }
    typedef BOOL (APIENTRY *PFN_SwapInterval)(int);
    PFN_SwapInterval swapInterval = (PFN_SwapInterval)wglGetProcAddress("wglSwapIntervalEXT");
    if (swapInterval) swapInterval(1);                    // sync to the monitor (no tearing)

    auto start = std::chrono::steady_clock::now();
    auto lastFrame = start, lastTitle = start;
    int frames = 0;

    while (g_running) {
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (!g_running) break;

        RECT rc; GetClientRect(hwnd, &rc);
        int cw = rc.right, ch = rc.bottom;
        if (cw < 8 || ch < 8) { Sleep(20); continue; }

        auto now = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(now - start).count();
        double dt   = std::chrono::duration<double>(now - lastFrame).count();
        lastFrame = now;
        if (dt > 0.1) dt = 0.1;
        if (g_hover) updateHoverCamera(hwnd, dt);                    // the cursor moves the point of view
        else if (g_autoSpin) orbitCamera(g_cam, 0.25 * dt, 0);

        drawScene(cw, ch, g_cam, time, g_showGrid, g_aa);
        SwapBuffers(hdc);

        ++frames;
        if (std::chrono::duration<double>(now - lastTitle).count() > 0.5) {
            double fps = frames / std::chrono::duration<double>(now - lastTitle).count();
            char title[220];
            snprintf(title, sizeof(title), "Black hole | %.0f fps | %s | move cursor = rotate view 360 | wheel=zoom | Z/X=sensitivity %.2f | M=mode R=reset SPACE=spin G=grid A=smooth",
                     fps, g_hover ? "HOVER view" : "DRAG view", g_sens);
            SetWindowTextA(hwnd, title);
            lastTitle = now; frames = 0;
        }
    }
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(ctx);
    return 0;
}