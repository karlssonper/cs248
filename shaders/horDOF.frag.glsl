uniform sampler2D phongTex;
uniform sampler2D cocTex;

uniform float texDx;
uniform float DOF;
varying vec2 texcoord;

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

vec3 gaussianBlur(sampler2D tex, float n)
{
    vec3 sum = vec3(0.0);
    float u = texcoord.x - n/2.0 * texDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, u += texDx){
        if (texture2D(cocTex,texcoord).r > 0.25) {
            float d = distance(texcoord, vec2(u,texcoord.y ));
            float weight = exp(-d*d);
            sum += weight*texture2D(tex, vec2(u,texcoord.y )).rgb;
            totWeight += weight;
        }

    }
    return sum / totWeight;
}

void main() {
    //First color buffer
    float blurAmount = texture2D(cocTex,texcoord).r;
    vec3 color;
    if (blurAmount > 0.25)
        color = gaussianBlur(phongTex,blurAmount*DOF);
    else
        color = texture2D(phongTex,texcoord).rgb;

	gl_FragData[0] = vec4(color,1.0);

	gl_FragData[1] = vec4(1);
}
