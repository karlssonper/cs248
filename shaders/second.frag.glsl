uniform sampler2D phongTex;
uniform sampler2D particlesTex;
uniform sampler2D bloomTex;
uniform sampler2D cocTex;
uniform sampler2D shadowTex;
uniform sampler2D depthTex;
uniform sampler2D hudTex;

uniform mat4 InverseViewProjection;
uniform mat4 PrevViewProjection;

uniform float debug;
varying vec2 texcoord;

uniform float texDx;

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

vec3 gaussianBlur(sampler2D tex, int n)
{
    vec3 sum = vec3(0.0);
    float v = texcoord.y - float(n)/2.0 * texDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, v += texDx){
        float d = distance(texcoord, vec2(texcoord.x,v ));
        float weight = exp(-d*d);
        sum += weight*texture2D(tex, vec2(texcoord.x,v)).rgb;
        totWeight += weight;
    }
    return sum / totWeight;
}

vec3 sceneColor(vec2 coords)
{
    vec3 bloom = texture2D(bloomTex, coords).rgb;
    vec4 part = texture2D(particlesTex, coords);
    vec3 phong = texture2D(phongTex, coords).rgb + 0.3*bloom;
    return (1.0-part.a) * phong + part.a*part.rgb;
}

vec3 motionBlur(float numSamples)
{
    float zOverW = texture2D(depthTex, texcoord).z;

    // H is the viewport position at this pixel in the range -1 to 1.
    vec4 H = vec4(
                   texcoord.x * 2.0 - 1.0,
                   (1.0 - texcoord.y) * 2.0 - 1.0,
                   zOverW,
                   1.0
                 );
    vec4 D = InverseViewProjection * H;
    vec4 worldPos = D / D.w;

    vec4 currentPos = H;
    vec4 previousPos = PrevViewProjection * worldPos;
    // Convert to nonhomogeneous points [-1,1] by dividing by w.
    previousPos /= previousPos.w;

    // Use this frame's position and last frame's to compute the pixel
    // velocity.
    vec2 velocity = (currentPos.xy - previousPos.xy)/2.0;

    // Get the initial color at this pixel.
    vec3 color = sceneColor(texcoord);
    texcoord += velocity;
    for(int i = 1; i < numSamples; ++i, texcoord += velocity)
    {
        // Sample the color buffer along the velocity vector.
        vec3 currentColor = sceneColor(texcoord);
        // Add the current color to our color sum.
        color += currentColor;
    }

    // Average all of the samples to get the final blur color.
    vec3 finalColor = color / numSamples;
    return finalColor;
}

void main() {
	vec3 color;
	if (debug == 1.0) {
	    vec4 hud = texture2D(hudTex, texcoord);
	    color = (1.0-hud.a)*motionBlur(2.25) + hud.a*(hud.rgb);
	} else if (debug == 2.0) {
	    color = gaussianBlur(bloomTex,10);
    } else if (debug == 3.0) {
        color = vec3(texture2D(particlesTex, texcoord).a);
        //color = texture2D(phongTex, texcoord) + gaussianBlur(10);
    } else if (debug == 4.0) {
        color = vec3(texture2D(cocTex, texcoord).r);
    } else if (debug == 5.0) {
        color = vec3(texture2D(shadowTex, texcoord).z);
    }

	//gl_FragColor = vec4(color,1);
	gl_FragData[0] = vec4(color,1);
	gl_FragData[1] = vec4(texture2D(cocTex,texcoord));
}
