uniform samplerCube skyboxTex;
uniform sampler2D shadowMap;

uniform float shadowMapDx;
uniform mat4 InverseViewMatrix;

varying vec3 eyePosition;
varying vec3 normal;
varying vec4 shadowcoord;
varying vec3 lightDir;

float shadow(vec2 tcoords, float depth)
{
    float distanceFromLight = texture2D(shadowMap,tcoords).z;
    return distanceFromLight < depth ? 0.25 : 1.0 ;
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

vec3 diffuse(vec3 L, vec3 N, vec3 V, vec3 diffuseRGB)
{
    // Calculate the diffuse color coefficient, and sample the diffuse texture
    vec3 Rnew = reflect(V, N);
    vec4 Rprim = InverseViewMatrix * vec4(Rnew,0);
    vec3 Td = textureCube(skyboxTex, Rprim.xyz).rgb;
    float Rd = max(0.0, dot(L, N));
    return Rd * (Td + diffuseRGB);
}

vec3 specular(vec3 L, vec3 N, vec3 V, vec3 specularRGB)
{
    // Calculate the specular coefficient
    vec3 R = reflect(-L, N);
    float Rs = pow(max(0.0, dot(V, R)), 120.0);
    return Rs * specularRGB;
}

float shadowScale(int n)
{
    vec4 shadowCoordinateWdivide = shadowcoord / shadowcoord.w;;
    shadowCoordinateWdivide.z -= 0.005;

    return gaussianShadow(shadowcoord.st,shadowCoordinateWdivide.z, n);
}

void main() {
    vec3 L = normalize(lightDir);
    vec3 N = normalize(normal);
    vec3 V = normalize(-eyePosition);

    vec3 diffuseColor = diffuse(L, N, V, vec3(0.05, 0.05, 0.15));
    vec3 specularColor = specular(L, N, V, vec3(0.6));
    vec3 ambientColor = (0.05, 0.05, 0.1);
    float ss = shadowScale(3);
    //Normal
    //gl_FragColor = vec4(0.5*N + vec3(0.5f,0.5f,0.5f),1);

    //Ambient
    //gl_FragColor = vec4(ambientColor,1);

    //Diffuse
    //gl_FragColor = vec4(diffuseColor,1);

    //Specular
    //gl_FragColor = vec4(specularColor,1);

    //Phong
    //gl_FragColor = vec4(ss*(ambientColor + diffuseColor  + specularColor), 1);

    //Depth
    gl_FragData[0] = vec4(1,1,1,1);

    //Phong Tex
    gl_FragData[1] = vec4(ss*(ambientColor + diffuseColor  + specularColor), 1);

    //Bloom Tex
    gl_FragData[2] = vec4(ambientColor + diffuseColor  + specularColor, 1);

    //Motion Tex
    gl_FragData[3] = vec4(ss,ss,ss,1);

    //CoC Tex
    gl_FragData[4] = vec4(0.5*normalize(N) + vec3(0.5),1);

}
