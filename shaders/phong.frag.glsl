uniform sampler2D normalMap;

varying vec2 texcoord;
varying vec3 eyePosition;
varying vec3 normal;
varying vec3 tangent;
varying vec3 bitangent;
varying vec3 L;

vec3 getNormal()
{
    vec3 SampledNormal = 2.0 * texture2D(normalMap, texcoord).rgb -
                         vec3(1.0, 1.0, 1.0);
    mat3 TBN = mat3(tangent,bitangent,normal);
    vec3 TangentNormal = TBN * SampledNormal;
    return normalize( TangentNormal);
}

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
	vec3 N = getNormal();
	vec3 V = normalize(-eyePosition);
	//vec3 L1 = normalize(gl_LightSource[0].position.xyz);
	//vec3 L2 = normalize(gl_LightSource[1].position.xyz - eyePosition);
	vec3 totDiffuse = diffuse(L, N, V, vec3(0.5, 0.5, 0.1));
    vec3 totSpecular = specular(L, N, V, vec3(1,1,1));
	vec3 totAmbient = 0.1;
	//gl_FragColor = vec4(totDiffuse+totAmbient+totSpecular, 1);
	//gl_FragColor = vec4(texcoord.x, 0,0,1);


	//Depth
	gl_FragData[0] = vec4(1,1,1,1);

	//Phong Tex
	gl_FragData[1] = vec4(0,1,0,1);

	//Bloom Tex
	gl_FragData[2] = vec4(0,0,1,1);

	//Motion Tex
	gl_FragData[3] = vec4(0.5*normalize(N) + vec3(0.5),1);

	//CoC Tex
	gl_FragData[4] = vec4(0.5*normalize(N) + vec3(0.5),1);
}
