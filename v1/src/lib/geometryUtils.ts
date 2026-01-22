import * as THREE from 'three';

/**
 * Calculates a circle from 3 points in 3D space.
 * Returns the Center point and the Normal vector of the circle's plane.
 */
export function calculateCircleFrom3Points(
    p1: THREE.Vector3,
    p2: THREE.Vector3,
    p3: THREE.Vector3
): { center: THREE.Vector3; normal: THREE.Vector3; radius: number } | null {
    // 1. Calculate two vectors
    const v1 = new THREE.Vector3().subVectors(p2, p1);
    const v2 = new THREE.Vector3().subVectors(p3, p1);

    // 2. Calculate the normal of the plane (cross product)
    const normal = new THREE.Vector3().crossVectors(v1, v2).normalize();

    // Check if points are collinear
    if (normal.lengthSq() === 0) return null;

    // 3. Find the center
    // We can use the intersection of perpendicular bisectors
    // Or a formula. Let's use a geometric approach suitable for 3D.
    // The center lies on the plane.

    // Midpoints
    const mid12 = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    const mid13 = new THREE.Vector3().addVectors(p1, p3).multiplyScalar(0.5);

    // Directions of bisectors (perpendicular to segments, in the plane)
    // BisectorDir1 = Normal x (p2 - p1)
    const bisectorDir1 = new THREE.Vector3().crossVectors(normal, v1).normalize();
    // BisectorDir2 = Normal x (p3 - p1)
    const bisectorDir2 = new THREE.Vector3().crossVectors(normal, v2).normalize();

    // Intersect two lines:
    // Line 1: mid12 + t * bisectorDir1
    // Line 2: mid13 + u * bisectorDir2

    // Solve for t:
    // mid12 + t*dir1 = mid13 + u*dir2
    // t*dir1 - u*dir2 = mid13 - mid12

    // This is a 2D system in the plane, but we can solve in 3D using dot products?
    // Easiest is to project to 2D, but let's try vector algebra.

    const delta = new THREE.Vector3().subVectors(mid13, mid12);
    // Project delta onto the plane defined by dir1 and dir2? They are in the same plane.

    // Cross product approach to solve linear system for intersection
    // From reliable algos:
    // Center = P1 + (v1_sq * (v1 x v2) x v2 + v2_sq * v1 x (v1 x v2)) / (2 * |v1 x v2|^2)
    // Ref: https://en.wikipedia.org/wiki/Circumcircle#Higher_dimensions

    // Let's use the explicit vector formula for circumcenter:
    // C = a + [ (||b-a||^2 ((c-a) x (b-a) x (b-a)) + ||c-a||^2 ((b-a) x (c-a) x (c-a)) ] / (2 || (b-a) x (c-a) ||^2)
    // where x is cross product.

    // Let a=p1, b=p2, c=p3
    // A = p2 - p1 (v1)
    // B = p3 - p1 (v2)
    // Cross = A x B

    const A = v1.clone();
    const B = v2.clone();
    const Cross = new THREE.Vector3().crossVectors(A, B);
    const CrossSq = Cross.lengthSq();

    if (CrossSq < 1e-10) return null; // Collinear

    // Term 1: A.lengthSq() * (Cross x A)
    const Term1 = new THREE.Vector3().crossVectors(Cross, A).multiplyScalar(B.lengthSq());

    // Term 2: B.lengthSq() * (B x Cross)  <-- Note order flip for sign correctness?
    // Formula: (A^2 * (B x (A x B)) + B^2 * ((A x B) x A)) / (2 * |AxB|^2)
    // Let's check commonly cited formula:
    // ( (A x B) x (A |B|^2 - B |A|^2) ) / (2 |A x B|^2) + P1

    // Let's use the robust one:
    // S = 2 * |Cross|^2
    // U = ( (Cross x A) * |B|^2 + (B x Cross) * |A|^2 ) / S
    // Center = P1 + U

    const ASq = A.lengthSq();
    const BSq = B.lengthSq();
    const S = 2 * CrossSq;

    const U1 = new THREE.Vector3().crossVectors(Cross, A).multiplyScalar(BSq);
    const U2 = new THREE.Vector3().crossVectors(B, Cross).multiplyScalar(ASq);

    const U = new THREE.Vector3().addVectors(U1, U2).divideScalar(S);

    const center = new THREE.Vector3().addVectors(p1, U);
    const radius = center.distanceTo(p1);

    return { center, normal, radius };
}
