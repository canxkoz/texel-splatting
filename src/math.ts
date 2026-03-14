export function perspective(
    fov: number,
    aspect: number,
    near: number,
    far: number,
): Float32Array<ArrayBuffer> {
    const out = new Float32Array(16);
    const f = 1 / Math.tan((fov * Math.PI) / 360);
    const nf = 1 / (near - far);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far * nf;
    out[11] = -1;
    out[14] = far * near * nf;
    return out;
}

export function lookAt(
    ex: number,
    ey: number,
    ez: number,
    tx: number,
    ty: number,
    tz: number,
    ux: number,
    uy: number,
    uz: number,
): Float32Array<ArrayBuffer> {
    let zx = ex - tx,
        zy = ey - ty,
        zz = ez - tz;
    const zLen = Math.sqrt(zx * zx + zy * zy + zz * zz);
    zx /= zLen;
    zy /= zLen;
    zz /= zLen;
    let xx = uy * zz - uz * zy;
    let xy = uz * zx - ux * zz;
    let xz = ux * zy - uy * zx;
    const xLen = Math.sqrt(xx * xx + xy * xy + xz * xz);
    xx /= xLen;
    xy /= xLen;
    xz /= xLen;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    const out = new Float32Array(16);
    out[0] = xx;
    out[1] = yx;
    out[2] = zx;
    out[4] = xy;
    out[5] = yy;
    out[6] = zy;
    out[8] = xz;
    out[9] = yz;
    out[10] = zz;
    out[12] = -(xx * ex + xy * ey + xz * ez);
    out[13] = -(yx * ex + yy * ey + yz * ez);
    out[14] = -(zx * ex + zy * ey + zz * ez);
    out[15] = 1;
    return out;
}

export function multiply(a: Float32Array, b: Float32Array): Float32Array<ArrayBuffer> {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            out[j * 4 + i] =
                a[i] * b[j * 4] +
                a[i + 4] * b[j * 4 + 1] +
                a[i + 8] * b[j * 4 + 2] +
                a[i + 12] * b[j * 4 + 3];
        }
    }
    return out;
}
