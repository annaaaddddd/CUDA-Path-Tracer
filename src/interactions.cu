#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Schlick approximation of the Fresnel reflectance for a dielectric
// cosTheta is the cosine between the incoming ray and the normal facing it
__host__ __device__ float schlickFresnel(float cosTheta, float ior)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 newDir;

    if (m.hasReflective > 0) {
        // Perfect specular: mirror the incoming ray about the surface normal
        newDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
    }
    else if (m.hasRefractive > 0) {
        // Dielectric (glass, water): reflect or refract, chosen by the Fresnel weight
        glm::vec3 dir = pathSegment.ray.direction;
        glm::vec3 n = normal;

        // entering when the ray travels against the outward normal;
        // when leaving, flip n so it faces the ray for the formulas below
        bool entering = dot(dir, n) < 0;
        if (!entering) n = -n;

        // eta = n1 / n2 as glm::refract expects (air -> glass on entry, glass -> air on exit)
        float eta = entering ? 1 / m.indexOfRefraction : m.indexOfRefraction;

        float cosI = -glm::dot(dir, n);
        float R = schlickFresnel(cosI, m.indexOfRefraction);

        // Snell discriminant; glm 0.9 refract returns NaN (not zero) on total internal
        // reflection, so the test has to happen here
        float k = 1 - (eta * eta) * (1 - cosI * cosI);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if (k < 0 || u01(rng) <  R) {
            newDir = glm::reflect(dir, n);
        }
        else {
            newDir = glm::refract(dir, n, eta);
        }
        // no division by the pick probability: choosing reflection with
        // probability R and weighting it by R cancels to 1
        pathSegment.color *= m.color;
    }
    else {
        // Ideal diffuse: cosine-weighted sampling makes bsdf * cos / pdf
        // collapse to just the albedo (cos and pi terms cancel)
        // normals are geometric (outward), so face the ray if hit from inside
        if (glm::dot(pathSegment.ray.direction, normal) > 0) normal = -normal;
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    pathSegment.ray.direction = normalize(newDir);
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction; // Offset origin off the surface to avoid self-intersection (shadow acne)
}
