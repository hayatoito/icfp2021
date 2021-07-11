use crate::prelude::*;

// Geometry

// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

pub enum IntersectResult {
    PointOnSegment,
    Intersect,
    None,
}

pub trait Intersect {
    fn intersect(&self, other: &Self) -> IntersectResult;
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
fn on_segment(p: Point, q: Point, r: Point) -> bool {
    q.0 <= p.0.max(r.0) && q.0 >= p.0.min(r.0) && q.1 <= p.1.max(r.1) && q.1 >= p.1.min(r.1)
}

fn orientation(p: Point, q: Point, r: Point) -> u32 {
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);

    if val == 0 {
        return 0; // colinear
    }

    // clock or counterclock wise
    if val > 0 {
        1
    } else {
        2
    }
}

// / The main function that returns true if line segment 'p1q1'
// // and 'p2q2' intersect.
// bool doIntersect(Point p1, Point q1, Point p2, Point q2)
impl Intersect for Segment {
    fn intersect(&self, other: &Segment) -> IntersectResult {
        let p1 = self.0;
        let q1 = self.1;

        let p2 = other.0;
        let q2 = other.1;
        // Find the four orientations needed for general and
        // special cases
        let o1 = orientation(p1, q1, p2);
        let o2 = orientation(p1, q1, q2);
        let o3 = orientation(p2, q2, p1);
        let o4 = orientation(p2, q2, q1);

        // println!("{} != {}, {} != {}", o1, o2, o3, o4);

        let on_boundary = || {
            // Special Cases
            // p1, q1 and p2 are colinear and p2 lies on segment p1q1
            if o1 == 0 && on_segment(p1, p2, q1) {
                return true;
            }

            // p1, q1 and q2 are colinear and q2 lies on segment p1q1
            if o2 == 0 && on_segment(p1, q2, q1) {
                return true;
            }

            // p2, q2 and p1 are colinear and p1 lies on segment p2q2
            if o3 == 0 && on_segment(p2, p1, q2) {
                return true;
            }

            // p2, q2 and q1 are colinear and q1 lies on segment p2q2
            if o4 == 0 && on_segment(p2, q1, q2) {
                return true;
            }

            false
        };

        // General case
        if o1 != o2 && o3 != o4 {
            if on_boundary() {
                return IntersectResult::PointOnSegment;
            }
            return IntersectResult::Intersect;
        }
        if on_boundary() {
            return IntersectResult::PointOnSegment;
        }
        return IntersectResult::None; // Doesn't fall in any of the above cases
    }
}

// https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
// Returns true if the point p lies inside the polygon[] with n vertices
pub fn is_inside(polygon: &[Point], p: Point) -> bool {
    // There must be at least 3 vertices in polygon[]
    assert!(polygon.len() >= 3);

    // Create a point for line segment from p to infinite
    let extreme = ((i32::MAX / 10) as i64, p.1);

    // Count intersections of the above line with sides of polygon
    let mut count = 0;
    for i in 0..polygon.len() {
        let next = (i + 1) % polygon.len();

        // Check if the line segment from 'p' to 'extreme' intersects
        // with the line segment from 'polygon[i]' to 'polygon[next]'
        // if (doIntersect(polygon[i], polygon[next], p, extreme))
        match (polygon[i], polygon[next]).intersect(&(p, extreme)) {
            // IntersectResult::Intersect => {
            //     count += 1;
            // }
            IntersectResult::Intersect | IntersectResult::PointOnSegment => {
                count += 1;
                // If the point 'p' is colinear with line segment 'i-next',
                // then check if it lies on segment. If it lies, return true,
                // otherwise false
                if orientation(polygon[i], p, polygon[next]) == 0 {
                    // debug!(
                    //     "p is on_segment: {:?} on {:?}",
                    //     p,
                    //     (polygon[i], polygon[next])
                    // );
                    return on_segment(polygon[i], p, polygon[next]);
                }
            }
            IntersectResult::None => {}
        }
    }

    // Return true if count is odd, false otherwise
    // if count % 2 == 0 {
    //     debug!("p is outside of polygon: {:?}, count: {}", p, count);
    // }
    count % 2 == 1
}

pub trait DistanceToSegment {
    fn distance_to_segment(&self, segment: &Segment) -> f64;
}

impl DistanceToSegment for Point {
    // https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
    fn distance_to_segment(&self, segment: &Segment) -> f64 {
        // // Function to return the minimum distance
        // // between a line segment AB and a point E
        // double minDistance(Point A, Point B, Point E)
        // {

        let e = P(*self);
        let a = P(segment.0);
        let b = P(segment.1);

        let ab = b - a;
        let be = e - b;
        let ae = e - a;

        // Variables to store dot product
        // double AB_BE, AB_AE;

        // // Calculating the dot product
        // AB_BE = (AB.F * BE.F + AB.S * BE.S);
        //     AB_AE = (AB.F * AE.F + AB.S * AE.S);

        let ab_be = ab.dot(&be);
        let ab_ae = ab.dot(&ae);

        // Minimum distance from
        // point E to the line segment

        if ab_be > 0 {
            // Case 1
            // Finding the magnitude
            // double y = E.S - B.S;
            // double x = E.F - B.F;
            // reqAns = sqrt(x * x + y * y);
            (e.squared_distance(&b) as f64).sqrt()
        } else if ab_ae < 0 {
            // Case 2
            // double y = E.S - A.S;
            // double x = E.F - A.F;
            // reqAns = sqrt(x * x + y * y);
            (e.squared_distance(&a) as f64).sqrt()
        } else {
            // Case 3
            // Finding the perpendicular distance
            // double x1 = AB.F;
            // double y1 = AB.S;
            // double x2 = AE.F;
            // double y2 = AE.S;
            // double mod = sqrt(x1 * x1 + y1 * y1);
            // reqAns = abs(x1 * y2 - y1 * x2) / mod;

            let x1 = ab.x();
            let y1 = ab.y();
            let x2 = ae.x();
            let y2 = ae.y();
            let m = ((x1 * x1 + y1 * y1) as f64).sqrt();
            (x1 * y2 - y1 * x2).abs() as f64 / m
        }
    }
}

#[cfg(test)]
mod tests {

    use geo::Coordinate;

    #[test]
    fn corddinate() {
        let p0 = Coordinate { x: 1, y: 2 };
        let p1 = Coordinate { x: 2, y: 1 };
        let p3 = p0 + p1;
        assert_eq!(p3.x_y(), (3, 3));
    }
}
