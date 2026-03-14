export const OKLAB_WGSL = /* wgsl */ `
fn hash2(p: vec2f) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn toOKLab(c: vec3f) -> vec3f {
    let lms = vec3(
        0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
        0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
        0.0883024619 * c.r + 0.2220049174 * c.g + 0.6896926207 * c.b,
    );
    let cbrt = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
    return vec3(
        0.2104542553 * cbrt.x + 0.7936177850 * cbrt.y - 0.0040720468 * cbrt.z,
        1.9779984951 * cbrt.x - 2.4285922050 * cbrt.y + 0.4505937099 * cbrt.z,
        0.0259040371 * cbrt.x + 0.7827717662 * cbrt.y - 0.8086757660 * cbrt.z,
    );
}

fn fromOKLab(lab: vec3f) -> vec3f {
    let l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    return max(vec3(
         4.0767416621 * l*l*l - 3.3077115913 * m*m*m + 0.2309699292 * s*s*s,
        -1.2684380046 * l*l*l + 2.6097574011 * m*m*m - 0.3413193965 * s*s*s,
        -0.0041960863 * l*l*l - 0.7034186147 * m*m*m + 1.7076147010 * s*s*s,
    ), vec3(0.0));
}

fn posterize(color: vec3f) -> vec3f {
    var lab = toOKLab(color);
    let L = clamp(lab.x, 0.0, 1.0);
    lab.x = floor(L * 32.0 + 0.5) / 32.0;
    lab.z += (lab.x - 0.5) * 0.05;
    return max(fromOKLab(lab), vec3f(0.0));
}
`;

export const PATH_WGSL = /* wgsl */ `
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

fn pathDist(fragXZ: vec2f) -> f32 {
    let lineDir = vec2(0.8, 0.6);
    let perpDir = vec2(-0.6, 0.8);
    let proj = dot(fragXZ, lineDir);
    let meander = sin(proj * 0.5) * 0.1;
    let closest = lineDir * clamp(proj, 20.0, 1e6) + perpDir * meander;
    let widthVar = 1.0 + value2d(vec2(proj * 0.3, 0.0), vec2(53.1, 97.3)) * 0.35;
    return length(fragXZ - closest) / widthVar;
}

fn pathGrassDiscard(fragXZ: vec2f) -> bool {
    let pd = pathDist(fragXZ);
    let grassNoise = value2d(fragXZ * 2.0, vec2(74.7, 173.3));
    let grassEdge = 0.8 + 0.4 + grassNoise * 0.3;
    let grassDither = hash2(floor(fragXZ * 15.0));
    return pd < grassEdge + (grassDither - 0.5) * 1.0;
}

fn pathGroundColor(fragXZ: vec2f, baseColor: vec3f) -> vec3f {
    let pd = pathDist(fragXZ);
    let dirtNoise = value2d(fragXZ * 2.0, vec2(127.1, 311.7));
    let dirtEdge = 0.8 + dirtNoise * 0.3;
    let dirtDither = hash2(floor(fragXZ * 20.0));
    let pathT = select(0.0, 1.0, pd < dirtEdge + (dirtDither - 0.5) * 0.5);

    let oklab = toOKLab(baseColor);
    let dirtL = oklab.x * 1.15;
    let dirtA = oklab.y * 0.3 + 0.01;
    let dirtB = oklab.z * 0.3 + 0.02;
    let dirt = fromOKLab(vec3(dirtL, dirtA, dirtB));
    return mix(baseColor, dirt, pathT);
}
`;
