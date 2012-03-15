varying vec3 eyePosition;
varying vec3 normal;
varying vec3 lightDir;

vec3 diffuse(vec3 L, vec3 N, vec3 V, vec3 diffuseRGB)
{
    // Calculate the diffuse color coefficient, and sample the diffuse texture
    float Rd = max(0.0, dot(L, N));
    return Rd * diffuseRGB;
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
	//gl_FragColor = vec4(0.5*L + vec3(0.5),1);
    //gl_FragColor = vec4(diffuse(L,N,-eyePosition,vec3(0.2,0.5,1)),1);
    gl_FragColor = vec4(0.5*N + vec3(0.5),1);

}
