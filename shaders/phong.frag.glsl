uniform sampler2D diffuseMap;
uniform sampler2D specularMap;
uniform sampler2D normalMap;
uniform sampler2D shadowMap;
uniform samplerCube cubeMap;

uniform vec3 Kd;
uniform vec3 Ks;
uniform vec3 Ka;
uniform float alpha;
uniform float hasDiffTex;
uniform float hasSpecTex;
uniform float hasNormalTex;
uniform float hasEnvTex;
uniform mat4 invView;
uniform float shadowMapDx;

varying vec2 texcoord;
varying vec3 normal;
varying vec3 eyePosition;
varying vec3 tangent;
varying vec3 bitangent;
varying vec4 shadowcoord;


float shadow(vec2 tcoords, float depth)
{
    float distanceFromLight = texture2D(shadowMap,tcoords).z;
    return distanceFromLight < depth ? 0.5 : 1.0 ;
}

float boxShadow(vec2 tcoords, float depth, int n)
{
    float sum;
    float u = tcoords.x - n/2.0 * shadowMapDx;
    float v = tcoords.y - n/2.0 * shadowMapDx;
    for (int i = 0; i < n; ++i, u += shadowMapDx){
        for (int j = 0; j < n; ++j, v += shadowMapDx){
            sum += shadow(vec2(u,v), depth);
            
        }
    }
    return sum / (n*n);
}

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

float gaussianShadow(vec2 tcoords, float depth, int n)
{
    float sum = 0.0;
    float u = tcoords.x - n/2.0 * shadowMapDx;
    float v = tcoords.y - n/2.0 * shadowMapDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, u += shadowMapDx){
        for (int j = 0; j < n; ++j, v += shadowMapDx){
            float d = distance(tcoords, vec2(u,v));
            float weight = exp(-d*d);
            sum += weight*shadow(vec2(u,v), depth);
            totWeight += weight;
        }
    }
    return sum / totWeight;
}

vec3 getNormal()
{
    if (hasNormalTex > 0.0){
       vec3 SampledNormal = 2.0 * texture2D(normalMap, texcoord).rgb - 
                            vec3(1.0, 1.0, 1.0);
       mat3 TBN = mat3(tangent,bitangent,normal);
       vec3 TangentNormal = TBN * SampledNormal;
       return normalize( TangentNormal);
    } else {
       return normalize(normal);
    }
}

vec3 diffuse(vec3 L, vec3 N, vec3 V, vec3 diffuseRGB)
{
    // Calculate the diffuse color coefficient, and sample the diffuse texture
    float Rd = max(0.0, dot(L, N));
    vec3 Td;
    if (hasDiffTex > 0.0) {
       Td = texture2D(diffuseMap, texcoord).rgb;
    } else {
       if (hasEnvTex > 0.0) {
           vec3 Rnew = reflect(V, N);
           vec4 Rprim = invView * vec4(Rnew,0);
           Td = textureCube(cubeMap, Rprim.xyz).rgb;
       } else {
           Td = vec3(1,1,1);
       }
    }
    return Rd * Kd * Td * diffuseRGB;
}

vec3 specular(vec3 L, vec3 N, vec3 V, vec3 specularRGB)
{
    // Calculate the specular coefficient
    vec3 R = reflect(-L, N);
    float Rs = pow(max(0.0, dot(V, R)), alpha);
    vec3 Ts;
    if (hasSpecTex > 0.0) {
       Ts = texture2D(specularMap, texcoord).rgb;
    } else {
       Ts = vec3(1,1,1);
    } 
    return Rs * Ks * Ts * specularRGB;
}

float shadowScale(int n)
{
    vec4 shadowCoordinateWdivide = shadowcoord / shadowcoord.w;;
    shadowCoordinateWdivide.z -= 0.005;

    return gaussianShadow(shadowcoord.st,shadowCoordinateWdivide.z, n);
}

void main() {
	vec3 N = getNormal();
	vec3 V = normalize(-eyePosition);
	vec3 L1 = normalize(gl_LightSource[0].position.xyz);
	vec3 L2 = normalize(gl_LightSource[1].position.xyz - eyePosition); 
	vec3 totDiffuse = diffuse(L1, N, V, gl_LightSource[0].diffuse.rgb) +
	                  diffuse(L2, N, V, gl_LightSource[1].diffuse.rgb);
    vec3 totSpecular = specular(L1, N, V, gl_LightSource[0].specular.rgb) +
                       specular(L2, N, V, gl_LightSource[1].specular.rgb);     	
	vec3 totAmbient = Ka * 
	   (gl_LightSource[0].ambient.rgb + gl_LightSource[1].ambient.rgb);
    float ss = shadowScale(3);
	gl_FragColor = vec4(ss*(totDiffuse+totAmbient+totSpecular), 1);
}
