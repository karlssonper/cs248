/*vec3 getNormal()
{
    return normalize(normal);
}

vec3 diffuse(vec3 L, vec3 N, vec3 V, vec3 diffuseRGB)
{
    // Calculate the diffuse color coefficient, and sample the diffuse texture
    float Rd = max(0.0, dot(L, N));
    return Rd * Kd * diffuseRGB;
}

vec3 specular(vec3 L, vec3 N, vec3 V, vec3 specularRGB)
{
    // Calculate the specular coefficient
    vec3 R = reflect(-L, N);
    float Rs = pow(max(0.0, dot(V, R)), 120.0);
    return Rs * Ks  * specularRGB;
}
*/
void main() {
	//vec3 N = getNormal();
	//vec3 V = normalize(-eyePosition);
	//vec3 L1 = normalize(gl_LightSource[0].position.xyz);
	//vec3 L2 = normalize(gl_LightSource[1].position.xyz - eyePosition);
	/*vec3 totDiffuse = diffuse(L1, N, V, gl_LightSource[0].diffuse.rgb) +
	                  diffuse(L2, N, V, gl_LightSource[1].diffuse.rgb);
    vec3 totSpecular = specular(L1, N, V, gl_LightSource[0].specular.rgb) +
                       specular(L2, N, V, gl_LightSource[1].specular.rgb);     	
	vec3 totAmbient = Ka * 
	   (gl_LightSource[0].ambient.rgb + gl_LightSource[1].ambient.rgb);
	gl_FragColor = vec4(totDiffuse+totAmbient+totSpecular, 1);*/
	gl_FragColor = vec4(1,1,1,1);
}
