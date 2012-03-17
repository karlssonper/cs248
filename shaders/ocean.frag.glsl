uniform samplerCube skyboxTex;

uniform mat4 InverseViewMatrix;

varying vec3 eyePosition;
varying vec3 normal;
varying vec3 lightDir;

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

void main() {
    vec3 L = normalize(lightDir);
    vec3 N = normalize(normal);
    vec3 V = normalize(-eyePosition);

    vec3 diffuseColor = diffuse(L, N, V, vec3(0.05, 0.05, 0.15));
    vec3 specularColor = specular(L, N, V, vec3(0.6));
    vec3 ambientColor = (0.05, 0.05, 0.1);

    //Normal
    //gl_FragColor = vec4(0.5*N + vec3(0.5f,0.5f,0.5f),1);

    //Ambient
    //gl_FragColor = vec4(ambientColor,1);

    //Diffuse
    //gl_FragColor = vec4(diffuseColor,1);

    //Specular
    //gl_FragColor = vec4(specularColor,1);

    //Phong
    gl_FragColor = vec4(ambientColor + diffuseColor  + specularColor, 1);

}
