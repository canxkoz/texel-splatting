interface ColorRGB {
    r: number;
    g: number;
    b: number;
}

interface GradientStop {
    elevation: number;
    sunColor: ColorRGB;
    sunIntensity: number;
    ambientColor: ColorRGB;
    ambientIntensity: number;
    zenith: ColorRGB;
    horizon: ColorRGB;
    sunDisk: number;
    sunGlow: number;
    starsIntensity: number;
    moonDisk: number;
    hazeDensity: number;
    hazeColor: ColorRGB;
    cloudsColor: ColorRGB;
}

export interface SkyOutput {
    sunColor: ColorRGB;
    sunIntensity: number;
    ambientColor: ColorRGB;
    ambientIntensity: number;
    zenith: ColorRGB;
    horizon: ColorRGB;
    sunDisk: number;
    sunGlow: number;
    starsIntensity: number;
    moonDisk: number;
    hazeDensity: number;
    hazeColor: ColorRGB;
    cloudsColor: ColorRGB;
}

const STOPS: GradientStop[] = [
    {
        elevation: -90,
        sunColor: { r: 0.25, g: 0.3, b: 0.55 },
        sunIntensity: 0.15,
        ambientColor: { r: 0.3, g: 0.25, b: 0.55 },
        ambientIntensity: 0.5,
        zenith: { r: 0.03, g: 0.02, b: 0.1 },
        horizon: { r: 0.05, g: 0.06, b: 0.12 },
        sunDisk: 0,
        sunGlow: 0,
        starsIntensity: 1.0,
        moonDisk: 1.0,
        hazeDensity: 0.001,
        hazeColor: { r: 0.05, g: 0.04, b: 0.12 },
        cloudsColor: { r: 0.1, g: 0.1, b: 0.18 },
    },
    {
        elevation: -18,
        sunColor: { r: 0.6, g: 0.2, b: 0.3 },
        sunIntensity: 0.1,
        ambientColor: { r: 0.25, g: 0.18, b: 0.45 },
        ambientIntensity: 0.5,
        zenith: { r: 0.05, g: 0.02, b: 0.15 },
        horizon: { r: 0.4, g: 0.15, b: 0.25 },
        sunDisk: 0,
        sunGlow: 0,
        starsIntensity: 0.6,
        moonDisk: 0.8,
        hazeDensity: 0.003,
        hazeColor: { r: 0.25, g: 0.1, b: 0.2 },
        cloudsColor: { r: 0.3, g: 0.12, b: 0.2 },
    },
    {
        elevation: 0,
        sunColor: { r: 1.0, g: 0.45, b: 0.15 },
        sunIntensity: 0.4,
        ambientColor: { r: 0.35, g: 0.2, b: 0.15 },
        ambientIntensity: 0.5,
        zenith: { r: 0.1, g: 0.08, b: 0.3 },
        horizon: { r: 0.95, g: 0.45, b: 0.15 },
        sunDisk: 0.7,
        sunGlow: 0.5,
        starsIntensity: 0.1,
        moonDisk: 0,
        hazeDensity: 0.008,
        hazeColor: { r: 0.85, g: 0.4, b: 0.15 },
        cloudsColor: { r: 0.95, g: 0.5, b: 0.2 },
    },
    {
        elevation: 8,
        sunColor: { r: 1.0, g: 0.7, b: 0.4 },
        sunIntensity: 0.7,
        ambientColor: { r: 0.5, g: 0.4, b: 0.3 },
        ambientIntensity: 0.7,
        zenith: { r: 0.15, g: 0.25, b: 0.55 },
        horizon: { r: 0.85, g: 0.6, b: 0.3 },
        sunDisk: 0.7,
        sunGlow: 0.3,
        starsIntensity: 0,
        moonDisk: 0,
        hazeDensity: 0.006,
        hazeColor: { r: 0.75, g: 0.55, b: 0.3 },
        cloudsColor: { r: 0.95, g: 0.75, b: 0.5 },
    },
    {
        elevation: 20,
        sunColor: { r: 1.0, g: 1.0, b: 1.0 },
        sunIntensity: 0.75,
        ambientColor: { r: 0.55, g: 0.52, b: 0.5 },
        ambientIntensity: 1.0,
        zenith: { r: 0.25, g: 0.48, b: 0.82 },
        horizon: { r: 0.52, g: 0.58, b: 0.68 },
        sunDisk: 0.7,
        sunGlow: 0.2,
        starsIntensity: 0,
        moonDisk: 0,
        hazeDensity: 0.005,
        hazeColor: { r: 0.48, g: 0.54, b: 0.64 },
        cloudsColor: { r: 1.0, g: 1.0, b: 1.0 },
    },
    {
        elevation: 50,
        sunColor: { r: 1.0, g: 1.0, b: 1.0 },
        sunIntensity: 0.81,
        ambientColor: { r: 0.53, g: 0.536, b: 0.54 },
        ambientIntensity: 1.0,
        zenith: { r: 0.25, g: 0.47, b: 0.815 },
        horizon: { r: 0.55, g: 0.61, b: 0.7 },
        sunDisk: 0.7,
        sunGlow: 0.15,
        starsIntensity: 0,
        moonDisk: 0,
        hazeDensity: 0.005,
        hazeColor: { r: 0.5, g: 0.56, b: 0.66 },
        cloudsColor: { r: 1.0, g: 1.0, b: 1.0 },
    },
];

function lerpColor(a: ColorRGB, b: ColorRGB, t: number): ColorRGB {
    return {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
    };
}

function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

export function sampleGradient(elevationDegrees: number): SkyOutput {
    const el = Math.max(-90, Math.min(90, elevationDegrees));

    let lo = 0;
    let hi = STOPS.length - 1;
    for (let i = 0; i < STOPS.length - 1; i++) {
        if (el >= STOPS[i].elevation && el <= STOPS[i + 1].elevation) {
            lo = i;
            hi = i + 1;
            break;
        }
    }

    if (el <= STOPS[0].elevation) {
        lo = 0;
        hi = 0;
    } else if (el >= STOPS[STOPS.length - 1].elevation) {
        lo = STOPS.length - 1;
        hi = STOPS.length - 1;
    }

    const a = STOPS[lo];
    const b = STOPS[hi];
    const range = b.elevation - a.elevation;
    const linear = range > 0 ? (el - a.elevation) / range : 0;
    const t = linear * linear * (3 - 2 * linear);

    return {
        sunColor: lerpColor(a.sunColor, b.sunColor, t),
        sunIntensity: lerp(a.sunIntensity, b.sunIntensity, t),
        ambientColor: lerpColor(a.ambientColor, b.ambientColor, t),
        ambientIntensity: lerp(a.ambientIntensity, b.ambientIntensity, t),
        zenith: lerpColor(a.zenith, b.zenith, t),
        horizon: lerpColor(a.horizon, b.horizon, t),
        sunDisk: lerp(a.sunDisk, b.sunDisk, t),
        sunGlow: lerp(a.sunGlow, b.sunGlow, t),
        starsIntensity: lerp(a.starsIntensity, b.starsIntensity, t),
        moonDisk: lerp(a.moonDisk, b.moonDisk, t),
        hazeDensity: lerp(a.hazeDensity, b.hazeDensity, t),
        hazeColor: lerpColor(a.hazeColor, b.hazeColor, t),
        cloudsColor: lerpColor(a.cloudsColor, b.cloudsColor, t),
    };
}

export function toDirection(azimuthDeg: number, elevationDeg: number): [number, number, number] {
    const az = (azimuthDeg * Math.PI) / 180;
    const el = (elevationDeg * Math.PI) / 180;
    const cosEl = Math.cos(el);
    return [-Math.sin(az) * cosEl, -Math.sin(el), -Math.cos(az) * cosEl];
}

function moonDirection(sunAzimuth: number, sunElevation: number): [number, number, number] {
    const moonAz = (sunAzimuth + 180) % 360;
    return toDirection(moonAz, Math.abs(sunElevation));
}

export function lightDirection(azimuth: number, elevation: number): [number, number, number] {
    if (elevation >= 0) return toDirection(azimuth, elevation);
    if (elevation <= -18) return moonDirection(azimuth, elevation);
    const linear = -elevation / 18;
    const t = linear * linear * (3 - 2 * linear);
    const [sx, sy, sz] = toDirection(azimuth, elevation);
    const [mx, my, mz] = moonDirection(azimuth, elevation);
    const bx = sx + (mx - sx) * t;
    const by = sy + (my - sy) * t;
    const bz = sz + (mz - sz) * t;
    const len = Math.hypot(bx, by, bz);
    if (len < 0.001) return moonDirection(azimuth, elevation);
    return [bx / len, by / len, bz / len];
}

export function shadowFade(elevation: number): number {
    const linear = Math.max(0, Math.min(1, (elevation + 0.5) / 1.5));
    return linear * linear * (3 - 2 * linear);
}

// Sky uniform layout (192 bytes):
//   0: hazeDensity + horizonBand + pad(8)   16: hazeColor
//  32: skyZenith                            48: skyHorizon
//  64: moonParams                           80: moonDirection
//  96: starParams                          112: cloudParams
// 128: cloudColor                          144: sunParams
// 160: sunVisualColor                      176: sunDirection
const skyData = new Float32Array(48);

export function uploadSky(
    device: GPUDevice,
    buffer: GPUBuffer,
    output: SkyOutput,
    azimuth: number,
    elevation: number,
): void {
    skyData[0] = output.hazeDensity;
    skyData[1] = 0; // horizonBand
    skyData[2] = 0;
    skyData[3] = 0;

    skyData[4] = output.hazeColor.r;
    skyData[5] = output.hazeColor.g;
    skyData[6] = output.hazeColor.b;
    skyData[7] = 1.0;

    skyData[8] = output.zenith.r;
    skyData[9] = output.zenith.g;
    skyData[10] = output.zenith.b;
    skyData[11] = 1.0; // active

    skyData[12] = output.horizon.r;
    skyData[13] = output.horizon.g;
    skyData[14] = output.horizon.b;
    skyData[15] = 1.0;

    // moonParams
    skyData[16] = 0.5; // phase
    skyData[17] = output.moonDisk; // opacity
    skyData[18] = output.moonDisk > 0 ? 1.0 : 0.0; // active
    skyData[19] = 0;

    // moonDirection (visual, pointing toward moon)
    const moonAz = ((azimuth + 180) % 360) * (Math.PI / 180);
    const moonEl = Math.abs(elevation) * (Math.PI / 180);
    const moonCosEl = Math.cos(moonEl);
    skyData[20] = Math.sin(moonAz) * moonCosEl;
    skyData[21] = Math.sin(moonEl);
    skyData[22] = Math.cos(moonAz) * moonCosEl;
    skyData[23] = 0;

    // starParams
    skyData[24] = output.starsIntensity; // intensity
    skyData[25] = 0.5; // amount
    skyData[26] = output.starsIntensity > 0 ? 1.0 : 0.0; // active
    skyData[27] = 0;

    // cloudParams
    skyData[28] = 0.9; // coverage
    skyData[29] = 0.8; // density
    skyData[30] = 0.5; // height
    skyData[31] = 1.0; // active

    // cloudColor
    skyData[32] = output.cloudsColor.r;
    skyData[33] = output.cloudsColor.g;
    skyData[34] = output.cloudsColor.b;
    skyData[35] = 0;

    // sunParams
    skyData[36] = output.sunDisk; // size
    skyData[37] = output.sunDisk > 0 ? 1.0 : 0.0; // active
    skyData[38] = 1.0; // colorOverride
    skyData[39] = output.sunGlow; // glow

    // sunVisualColor (warm white)
    skyData[40] = 1.0;
    skyData[41] = 0.94;
    skyData[42] = 0.85;
    skyData[43] = 0;

    // sunDirection (visual, pointing toward sun)
    const sunAz = azimuth * (Math.PI / 180);
    const sunEl = elevation * (Math.PI / 180);
    const sunCosEl = Math.cos(sunEl);
    skyData[44] = Math.sin(sunAz) * sunCosEl;
    skyData[45] = Math.sin(sunEl);
    skyData[46] = Math.cos(sunAz) * sunCosEl;
    skyData[47] = 0;

    device.queue.writeBuffer(buffer, 0, skyData);
}

// SkyScene uniform layout (128 bytes):
//   0: cameraWorld (mat4x4, 64 bytes)
//  64: sunDirection(12)+pad(4)   80: sunColor(12)+pad(4)
//  96: clearColor(12)+pad(4)    112: viewport(8)+fov(4)+cameraMode(4)
const sceneData = new Float32Array(32);

export function uploadSkyScene(
    device: GPUDevice,
    buffer: GPUBuffer,
    cameraWorld: Float32Array,
    sunDir: [number, number, number],
    sunColor: [number, number, number],
    clearColor: ColorRGB,
    width: number,
    height: number,
    fov: number,
): void {
    sceneData.set(cameraWorld, 0);
    sceneData[16] = sunDir[0];
    sceneData[17] = sunDir[1];
    sceneData[18] = sunDir[2];
    sceneData[19] = 0;
    sceneData[20] = sunColor[0];
    sceneData[21] = sunColor[1];
    sceneData[22] = sunColor[2];
    sceneData[23] = 1;
    sceneData[24] = clearColor.r;
    sceneData[25] = clearColor.g;
    sceneData[26] = clearColor.b;
    sceneData[27] = 1;
    sceneData[28] = width;
    sceneData[29] = height;
    sceneData[30] = fov;
    sceneData[31] = 0; // perspective
    device.queue.writeBuffer(buffer, 0, sceneData);
}

export const SKY_STRUCT_WGSL = /* wgsl */ `
struct Sky {
    hazeDensity: f32,
    horizonBand: f32,
    _pad3: f32,
    _pad4: f32,
    hazeColor: vec4<f32>,
    skyZenith: vec4<f32>,
    skyHorizon: vec4<f32>,
    moonParams: vec4<f32>,
    moonDirection: vec4<f32>,
    starParams: vec4<f32>,
    cloudParams: vec4<f32>,
    cloudColor: vec4<f32>,
    sunParams: vec4<f32>,
    sunVisualColor: vec4<f32>,
    sunDirection: vec4<f32>,
}`;

export const HAZE_WGSL = /* wgsl */ `
fn applyHaze(color: vec3<f32>, dist: f32) -> vec3<f32> {
    if (sky.hazeDensity <= 0.0) {
        return color;
    }
    let haze = 1.0 - exp(-sky.hazeDensity * dist);
    return mix(color, sky.hazeColor.rgb, haze);
}`;

export const NOISE_WGSL = /* wgsl */ `
fn hash2sky(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn value2d(p: vec2f, seed: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(fract(sin(dot(i, seed)) * 43758.5) * 2.0 - 1.0,
            fract(sin(dot(i + vec2(1.0, 0.0), seed)) * 43758.5) * 2.0 - 1.0, u.x),
        mix(fract(sin(dot(i + vec2(0.0, 1.0), seed)) * 43758.5) * 2.0 - 1.0,
            fract(sin(dot(i + vec2(1.0, 1.0), seed)) * 43758.5) * 2.0 - 1.0, u.x), u.y);
}

fn simplex2(p: vec2<f32>) -> f32 {
    let K1 = 0.366025404;
    let K2 = 0.211324865;

    let i = floor(p + (p.x + p.y) * K1);
    let a = p - i + (i.x + i.y) * K2;

    let o = select(vec2(0.0, 1.0), vec2(1.0, 0.0), a.x > a.y);
    let b = a - o + K2;
    let c = a - 1.0 + 2.0 * K2;

    let h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), vec3(0.0));
    let h4 = h * h * h * h;

    let n = vec3(
        dot(a, vec2(hash2sky(i) * 2.0 - 1.0, hash2sky(i + vec2(0.0, 1.0)) * 2.0 - 1.0)),
        dot(b, vec2(hash2sky(i + o) * 2.0 - 1.0, hash2sky(i + o + vec2(0.0, 1.0)) * 2.0 - 1.0)),
        dot(c, vec2(hash2sky(i + 1.0) * 2.0 - 1.0, hash2sky(i + vec2(1.0, 2.0)) * 2.0 - 1.0))
    );

    return dot(h4, n) * 70.0;
}

const FBM2_OCTAVES = 5;

fn fbm2(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < FBM2_OCTAVES; i++) {
        value += amplitude * simplex2(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}`;

export const STARS_WGSL = /* wgsl */ `
fn hashStar(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash2Star(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3(p.x, p.y, p.x) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn sampleStars(dir: vec3<f32>) -> vec3<f32> {
    if (sky.starParams.z <= 0.0 || dir.y < 0.0) {
        return vec3(0.0);
    }

    let theta = atan2(dir.z, dir.x);
    let phi = asin(clamp(dir.y, -1.0, 1.0));

    let gridSize = mix(20.0, 100.0, sky.starParams.y);
    let cell = vec2(theta * gridSize / 3.14159, phi * gridSize / 1.5708);
    let cellId = floor(cell);
    let cellFract = fract(cell);

    var starColor = vec3(0.0);

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = cellId + vec2(f32(dx), f32(dy));
            let starHash = hashStar(neighbor);

            if (starHash > sky.starParams.y * 0.7) {
                continue;
            }

            let starPos = hash2Star(neighbor);
            let starCenter = neighbor + starPos;
            let dist = length(cell - starCenter);

            let brightness = hashStar(neighbor + vec2(100.0, 100.0));
            let radius = 0.02 + brightness * 0.03;

            if (dist < radius) {
                let twinkle = 0.8 + 0.2 * sin(brightness * 100.0);
                let intensity = sky.starParams.x * brightness * twinkle;
                let falloff = 1.0 - smoothstep(0.0, radius, dist);

                let temp = hashStar(neighbor + vec2(200.0, 200.0));
                let tint = mix(vec3(1.0, 0.9, 0.8), vec3(0.8, 0.9, 1.0), temp);

                starColor = max(starColor, tint * intensity * falloff);
            }
        }
    }

    return starColor;
}`;

export const MOON_WGSL = /* wgsl */ `
fn sampleMoon(dir: vec3<f32>) -> vec3<f32> {
    if (sky.moonParams.z <= 0.0) {
        return vec3(0.0);
    }

    let moonDir = sky.moonDirection.xyz;
    let moonDot = dot(dir, moonDir);

    let moonSize = 0.9995;
    let moonColor = vec3(0.9, 0.9, 0.85);
    let edgeWidth = 0.0003;
    let opacity = sky.moonParams.y;

    if (moonDot <= moonSize - edgeWidth) {
        return vec3(0.0);
    }

    let toCenter = dir - moonDir * moonDot;
    let diskRight = normalize(cross(moonDir, vec3(0.0, 1.0, 0.0)));
    let diskUp = cross(diskRight, moonDir);

    let diskRadius = sqrt(1.0 - moonSize * moonSize);
    let u = dot(toCenter, diskRight) / diskRadius;
    let v = dot(toCenter, diskUp) / diskRadius;

    let r2 = u * u + v * v;
    let z = sqrt(max(0.0, 1.0 - r2));

    let diskEdge = smoothstep(1.0 + edgeWidth / diskRadius, 1.0 - edgeWidth / diskRadius, sqrt(r2));

    let limb = pow(z, 0.6);

    let cellU = u * 8.0;
    let cellV = v * 8.0;
    let craterNoise = hashStar(floor(vec2(cellU, cellV)) + vec2(50.0, 50.0));
    let surfaceVariation = 0.85 + 0.15 * craterNoise;

    let phase = sky.moonParams.x;
    let sunAngle = phase * 6.28318;
    let sunLocalX = sin(sunAngle);
    let sunLocalZ = -cos(sunAngle);

    let illumination = u * sunLocalX + z * sunLocalZ;
    let lit = smoothstep(-0.05, 0.05, illumination);

    let earthshine = vec3(0.06, 0.07, 0.1);
    let dayColor = moonColor * surfaceVariation * limb;
    let surfaceColor = mix(earthshine * limb, dayColor, lit);

    return surfaceColor * diskEdge * opacity;
}`;

export const CLOUDS_WGSL = /* wgsl */ `
fn sampleClouds(dir: vec3<f32>) -> vec4<f32> {
    if (sky.cloudParams.w <= 0.0 || dir.y < 0.01) {
        return vec4(0.0);
    }

    let t = sky.cloudParams.z / max(dir.y, 0.001);
    let uv = dir.xz * t;

    var n = fbm2(uv);

    let coverage = sky.cloudParams.x;
    let density = sky.cloudParams.y;
    n = smoothstep(1.0 - coverage, 1.0, n * 0.5 + 0.5) * density;

    n *= smoothstep(0.0, 0.15, dir.y);

    return vec4(sky.cloudColor.rgb, n);
}`;

export const SAMPLE_SKY_WGSL = /* wgsl */ `
fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    if (sky.skyZenith.a <= 0.0) {
        return scene.clearColor.rgb;
    }

    let t = pow(clamp(dir.y, 0.0, 1.0), 0.25);
    var color = mix(sky.skyHorizon.rgb, sky.skyZenith.rgb, t);

    if (sky.horizonBand > 0.0) {
        let horizonBlend = pow(1.0 - abs(dir.y), 32.0) * sky.horizonBand;
        let bandColor = sky.skyHorizon.rgb * 1.5;
        color = mix(color, bandColor, horizonBlend);
    }

    color += sampleStars(dir);

    let clouds = sampleClouds(dir);
    color = mix(color, clouds.rgb, clouds.a);

    let moonContrib = sampleMoon(dir);
    color += moonContrib * (1.0 - clouds.a * 0.7);

    if (sky.sunParams.y > 0.0) {
        let sunDir = sky.sunDirection.xyz;
        let sunDot = dot(dir, sunDir);

        let sunVisualColor = select(scene.sunColor.rgb, sky.sunVisualColor.rgb, sky.sunParams.z > 0.5);

        let glowStrength = sky.sunParams.w;
        if (glowStrength > 0.0) {
            let g = 0.76;
            let gg = g * g;
            let mie = (1.0 - gg) / pow(1.0 + gg - 2.0 * g * sunDot, 1.5);
            color += sunVisualColor * mie * glowStrength * 0.02;

            let angle = max(0.0, sunDot);
            let corona = pow(angle, 512.0) * 0.4 + pow(angle, 128.0) * 0.06;
            let warmTint = vec3f(1.0, 0.9, 0.7);
            color += warmTint * sunVisualColor * corona * glowStrength;
        }

        let baseSunSize = 0.9995;
        let sunSizeParam = sky.sunParams.x;
        let sunThreshold = 1.0 - (1.0 - baseSunSize) * sunSizeParam;
        let sunEdgeWidth = (1.0 - sunThreshold) * 0.15;

        let diskBlend = smoothstep(sunThreshold - sunEdgeWidth, sunThreshold + sunEdgeWidth, sunDot);
        if (diskBlend > 0.0) {
            let radial = saturate((sunDot - sunThreshold) / (1.0 - sunThreshold));
            let r = 1.0 - radial;
            let mu = sqrt(1.0 - r * r);
            let limbDarken = 1.0 - 0.6 * (1.0 - mu);
            color += sunVisualColor * limbDarken * diskBlend;

            let edgeDist = 1.0 - smoothstep(0.0, 1.0, radial);
            let fringe = vec3f(
                smoothstep(0.3, 0.7, edgeDist),
                smoothstep(0.5, 0.9, edgeDist),
                smoothstep(0.7, 1.0, edgeDist)
            );
            color += fringe * sunVisualColor * 0.15 * diskBlend * (1.0 - radial);
        }
    }

    if (sky.hazeDensity > 0.0) {
        let horizonFactor = 1.0 - clamp(dir.y, 0.0, 1.0);
        let hazeAmount = pow(horizonFactor, 2.0) * saturate(sky.hazeDensity * 5.0);
        color = mix(color, sky.hazeColor.rgb, hazeAmount);
    }

    return color;
}`;

export const SKY_SCENE_STRUCT_WGSL = /* wgsl */ `
struct SkyScene {
    cameraWorld: mat4x4<f32>,
    sunDirection: vec4<f32>,
    sunColor: vec4<f32>,
    clearColor: vec4<f32>,
    viewport: vec2<f32>,
    fov: f32,
    cameraMode: f32,
}`;

export const COMPUTE_SKY_DIR_WGSL = /* wgsl */ `
const DEG_TO_RAD: f32 = 0.017453292;

fn computeSkyDir(screenX: f32, screenY: f32) -> vec3<f32> {
    let width = scene.viewport.x;
    let height = scene.viewport.y;

    let ndcX = screenX * 2.0 - 1.0;
    let ndcY = 1.0 - screenY * 2.0;

    let aspect = width / height;

    let cameraWorld = scene.cameraWorld;
    let r00 = cameraWorld[0][0]; let r10 = cameraWorld[0][1]; let r20 = cameraWorld[0][2];
    let r01 = cameraWorld[1][0]; let r11 = cameraWorld[1][1]; let r21 = cameraWorld[1][2];
    let r02 = cameraWorld[2][0]; let r12 = cameraWorld[2][1]; let r22 = cameraWorld[2][2];

    let skyFov = select(scene.fov, 60.0, scene.cameraMode > 0.5);
    let tanHalfFov = tan((skyFov * DEG_TO_RAD) / 2.0);
    let camDirX = ndcX * aspect * tanHalfFov;
    let camDirY = ndcY * tanHalfFov;
    let camDirZ = -1.0;
    var dirX = r00 * camDirX + r01 * camDirY + r02 * camDirZ;
    var dirY = r10 * camDirX + r11 * camDirY + r12 * camDirZ;
    var dirZ = r20 * camDirX + r21 * camDirY + r22 * camDirZ;
    let len = sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
    dirX /= len; dirY /= len; dirZ /= len;
    return vec3(dirX, dirY, dirZ);
}`;

const SKY_SHADER = /* wgsl */ `
${SKY_SCENE_STRUCT_WGSL}

${SKY_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> scene: SkyScene;
@group(0) @binding(1) var<uniform> sky: Sky;

${COMPUTE_SKY_DIR_WGSL}

${NOISE_WGSL}
${STARS_WGSL}
${MOON_WGSL}
${CLOUDS_WGSL}
${SAMPLE_SKY_WGSL}

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    let positions = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    let p = positions[vi];
    var out: VsOut;
    out.pos = vec4f(p, 0, 1);
    out.uv = p * vec2f(0.5, -0.5) + 0.5;
    return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let dir = computeSkyDir(in.uv.x, in.uv.y);
    let color = sampleSky(dir);
    return vec4f(color, 1);
}
`;

export function createSky(device: GPUDevice, format: GPUTextureFormat) {
    const sceneBuffer = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const skyBuffer = device.createBuffer({
        size: 192,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = device.createShaderModule({ code: SKY_SHADER });

    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module },
        fragment: { module, targets: [{ format }] },
        primitive: { topology: "triangle-list" },
        depthStencil: {
            format: "depth32float",
            depthWriteEnabled: false,
            depthCompare: "always",
        },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sceneBuffer } },
            { binding: 1, resource: { buffer: skyBuffer } },
        ],
    });

    return {
        pipeline,
        sceneBuffer,
        skyBuffer,
        bindGroup,
        encode(encoder: GPUCommandEncoder, colorView: GPUTextureView, depthView: GPUTextureView) {
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: colorView,
                        clearValue: [0, 0, 0, 1],
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
                depthStencilAttachment: {
                    view: depthView,
                    depthClearValue: 1,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(3);
            pass.end();
        },
    };
}
