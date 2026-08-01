//! Connected components and pixel-domain shape measurements for binary masks.

/// Neighborhood used to join non-zero mask pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Connectivity {
    Four,
    #[default]
    Eight,
}

/// One connected set of non-zero pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryComponent {
    /// Row-major indices into the source mask.
    pub pixels: Vec<usize>,
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
    pub centroid_x: f32,
    pub centroid_y: f32,
    /// Rotation-independent ratio between the major and minor spread axes.
    pub elongation: f32,
}

impl BinaryComponent {
    pub fn width(&self) -> usize {
        self.max_x - self.min_x + 1
    }

    pub fn height(&self) -> usize {
        self.max_y - self.min_y + 1
    }

    pub fn fill_fraction(&self) -> f32 {
        self.pixels.len() as f32 / (self.width() * self.height()) as f32
    }
}

/// Find all connected non-zero regions in a row-major binary mask.
///
/// Components follow first-seen row-major order. This keeps output stable when
/// two components have the same size.
pub fn connected_components(
    mask: &[u8],
    width: usize,
    height: usize,
    connectivity: Connectivity,
) -> Vec<BinaryComponent> {
    let mut output = Vec::new();
    visit_components(mask, width, height, connectivity, |pixels| {
        output.push(measure_component(pixels, width));
    });
    output
}

/// Find the largest connected non-zero region without retaining smaller ones.
///
/// Ties resolve to the first component in row-major discovery order.
pub fn largest_connected_component(
    mask: &[u8],
    width: usize,
    height: usize,
    connectivity: Connectivity,
) -> Option<BinaryComponent> {
    let mut largest = None;
    visit_components(mask, width, height, connectivity, |pixels| {
        if largest
            .as_ref()
            .is_none_or(|component: &BinaryComponent| pixels.len() > component.pixels.len())
        {
            largest = Some(measure_component(pixels, width));
        }
    });
    largest
}

fn visit_components(
    mask: &[u8],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    mut visit: impl FnMut(Vec<usize>),
) {
    assert_eq!(mask.len(), width * height);
    if width == 0 || height == 0 {
        return;
    }
    let mut visited = vec![false; mask.len()];
    for start in 0..mask.len() {
        if visited[start] || mask[start] == 0 {
            continue;
        }
        visited[start] = true;
        let mut pending = vec![start];
        let mut pixels = Vec::new();
        while let Some(index) = pending.pop() {
            let x = index % width;
            let y = index / width;
            pixels.push(index);
            let x_start = x.saturating_sub(1);
            let x_end = (x + 1).min(width - 1);
            let y_start = y.saturating_sub(1);
            let y_end = (y + 1).min(height - 1);
            for neighbor_y in y_start..=y_end {
                for neighbor_x in x_start..=x_end {
                    if connectivity == Connectivity::Four && neighbor_x != x && neighbor_y != y {
                        continue;
                    }
                    let neighbor = neighbor_y * width + neighbor_x;
                    if !visited[neighbor] && mask[neighbor] != 0 {
                        visited[neighbor] = true;
                        pending.push(neighbor);
                    }
                }
            }
        }
        visit(pixels);
    }
}

fn measure_component(pixels: Vec<usize>, width: usize) -> BinaryComponent {
    let mut min_x = usize::MAX;
    let mut max_x = 0;
    let mut min_y = usize::MAX;
    let mut max_y = 0;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    for &index in &pixels {
        let x = index % width;
        let y = index / width;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        sum_x += x as f32;
        sum_y += y as f32;
    }
    let count = pixels.len() as f32;
    let centroid_x = sum_x / count;
    let centroid_y = sum_y / count;
    let mut covariance_xx = 0.0;
    let mut covariance_xy = 0.0;
    let mut covariance_yy = 0.0;
    for &index in &pixels {
        let dx = (index % width) as f32 - centroid_x;
        let dy = (index / width) as f32 - centroid_y;
        covariance_xx += dx * dx;
        covariance_xy += dx * dy;
        covariance_yy += dy * dy;
    }
    covariance_xx /= count;
    covariance_xy /= count;
    covariance_yy /= count;
    let half_trace = (covariance_xx + covariance_yy) * 0.5;
    let root =
        (((covariance_xx - covariance_yy) * 0.5).powi(2) + covariance_xy * covariance_xy).sqrt();
    let major = half_trace + root;
    let minor = (half_trace - root).max(0.0);
    // A quarter-pixel floor keeps a one-pixel-wide line finite while retaining
    // a useful large ratio for thin features.
    let elongation = ((major + 0.25) / (minor + 0.25)).sqrt();
    BinaryComponent {
        pixels,
        min_x,
        max_x,
        min_y,
        max_y,
        centroid_x,
        centroid_y,
        elongation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_connectivity_joins_a_diagonal() {
        let mask = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        let components = connected_components(&mask, 3, 3, Connectivity::Eight);

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].pixels.len(), 3);
        assert!(components[0].elongation > 2.0);
    }

    #[test]
    fn four_connectivity_keeps_diagonal_pixels_apart() {
        let mask = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        let components = connected_components(&mask, 3, 3, Connectivity::Four);

        assert_eq!(components.len(), 3);
    }

    #[test]
    fn measures_bounds_centroid_and_fill() {
        let mut mask = vec![0; 5 * 4];
        for y in 1..=2 {
            for x in 1..=3 {
                mask[y * 5 + x] = 1;
            }
        }
        let component = connected_components(&mask, 5, 4, Connectivity::Eight)
            .pop()
            .unwrap();

        assert_eq!((component.min_x, component.max_x), (1, 3));
        assert_eq!((component.min_y, component.max_y), (1, 2));
        assert_eq!((component.centroid_x, component.centroid_y), (2.0, 1.5));
        assert_eq!(component.fill_fraction(), 1.0);
    }

    #[test]
    fn largest_component_does_not_replace_an_equal_first_match() {
        let mask = [1, 1, 0, 1, 1];
        let component = largest_connected_component(&mask, 5, 1, Connectivity::Eight).unwrap();

        assert_eq!(component.pixels, vec![0, 1]);
    }
}
