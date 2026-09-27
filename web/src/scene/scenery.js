// Street furniture and planting that camera.yaml does not describe, traced by hand on a gridded copy of the
// C3905 reference frame (frame 0) in normalized image coordinates. Decoration only: nothing here is used by
// the pipeline or by any number shown on the site. Road, median, islands, crossings, stop line, sidewalks
// and the signal position come from configs/camera.yaml via scene.json.

export const SCENERY = {
  buildings: [
    { poly: [[0.13, 0.0], [0.30, 0.0], [0.30, 0.055], [0.13, 0.05]], fins: 0, floors: 1 },
    { poly: [[0.33, 0.0], [0.655, 0.0], [0.655, 0.125], [0.33, 0.095]], fins: 11, floors: 2 },
    { poly: [[0.655, 0.0], [1.0, 0.0], [1.0, 0.235], [0.655, 0.14]], fins: 0, floors: 3, glass: true },
  ],
  grass: [
    [[0.695, 0.285], [0.94, 0.33], [0.945, 0.37], [0.85, 0.372], [0.72, 0.318]],
    [[0.30, 0.085], [0.44, 0.1], [0.44, 0.115], [0.30, 0.1]],
  ],
  // trunk base (x, y), canopy radius as a fraction of frame width, canopy lift above the base
  trees: [
    { x: 0.034, y: 0.39, r: 0.06, lift: 0.26, big: true },
    { x: 0.245, y: 0.115, r: 0.035, lift: 0.1 },
    { x: 0.325, y: 0.108, r: 0.04, lift: 0.1 },
    { x: 0.395, y: 0.1, r: 0.038, lift: 0.09 },
    { x: 0.555, y: 0.14, r: 0.04, lift: 0.11 },
    { x: 0.672, y: 0.265, r: 0.055, lift: 0.2 },
    { x: 0.705, y: 0.285, r: 0.05, lift: 0.21 },
    { x: 0.73, y: 0.29, r: 0.05, lift: 0.22 },
    { x: 0.785, y: 0.33, r: 0.058, lift: 0.24 },
    { x: 0.857, y: 0.355, r: 0.06, lift: 0.26 },
    { x: 0.93, y: 0.33, r: 0.055, lift: 0.25 },
  ],
  busStop: { x0: 0.468, x1: 0.522, base: 0.145, top: 0.112, label: [0.494, 0.098] },
  gantry: { posts: [[0.106, 0.655], [0.127, 0.648]], top: 0.3, arm: [[0.1, 0.302], [0.39, 0.266]],
    heads: [[0.212, 0.285], [0.385, 0.262]], sign: [0.302, 0.3] },
  lamp: { base: [0.109, 0.62], top: [0.109, 0.028], panel: [0.092, 0.012, 0.125, 0.04] },
  medianPole: { base: [0.598, 0.478], top: [0.598, 0.33] },
  crossingSign: { base: [0.64, 0.49], top: [0.64, 0.325] },
  roundSign: { base: [0.672, 0.505], top: [0.672, 0.462] },
  bollards: [[0.143, 0.55], [0.155, 0.565], [0.161, 0.585]],
  bins: [[0.052, 0.74], [0.078, 0.705]],
  cracks: [
    [[0.52, 0.92], [0.55, 0.88], [0.6, 0.87], [0.63, 0.9]],
    [[0.66, 0.76], [0.7, 0.74], [0.72, 0.7]],
    [[0.78, 0.96], [0.8, 0.9], [0.85, 0.88]],
    [[0.46, 0.66], [0.49, 0.68], [0.5, 0.72]],
  ],
  places: [
    { text: 'bus stop', at: [0.494, 0.085] },
    { text: 'median signal', at: [0.575, 0.305], align: 'right' },
    { text: 'left plaza', at: [0.03, 0.6] },
    { text: 'near crossing', at: [0.33, 0.585], rot: -0.2 },
    { text: 'far crossing', at: [0.84, 0.445], rot: -0.2 },
    { text: 'refuge', at: [0.33, 0.765] },
    { text: 'to the junction', at: [0.76, 0.66] },
  ],
};

// Stripe direction of each zebra (the traffic direction it crosses) comes from the camera.yaml approaches;
// the stripe count is a drawing choice.
export const ZEBRA = {
  cross_near: { approach: 'near', stripes: 26 },
  cross_far: { approach: 'far', stripes: 20 },
  cross_foreground: { approach: 'through_exit', stripes: 22 },
};
