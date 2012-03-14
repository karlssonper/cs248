varying vec3 eyePosition;
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
	gl_FragColor = vec4(1,1,1,1);

}
