import { NextFunction, Request, Response } from "express"
import jwt, { JwtPayload } from "jsonwebtoken"

const verifyHospitalToken = async (req: Request, res: Response, next: NextFunction) => {
    const { type } = req.body
    const { h_token: token } = req.body

    if (!token) return res.status(401).json({ message: 'Unauthorized: No token provided' });
    if (!type || type !== 'H') return res.status(417).send("Unexpected body.")

    const { uid  } = req.body

    try {
        const d: JwtPayload | string = jwt.verify(token, process.env.HOSPITAL_SECRET || '');
        if (typeof d === 'object' && (d?.uid !== uid || d?.type !== type)) throw new Error("Not Authorized.")
            if (typeof d === 'object') req.body.hospital_id = d.hospital_id
            next();
    } catch (err) {
        return res.status(403).json({ message: 'Forbidden: Invalid token', err });
    }
}

export default verifyHospitalToken
