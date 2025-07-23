#!/usr/bin/env python3
"""
YouTube Frame Extractor
Télécharge des vidéos YouTube et extrait les frames localement
"""

import os
import sys
import json
import cv2
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import yt_dlp
from tqdm import tqdm
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_extractor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Structure des métadonnées vidéo"""
    video_id: str
    title: str
    description: str
    duration: int
    upload_date: str
    uploader: str
    view_count: int
    like_count: int
    url: str
    file_size: int
    resolution: str
    fps: float
    format: str
    extracted_frames: int
    download_date: str

class YouTubeFrameExtractor:
    def __init__(self, output_dir: str = "./youtube_data", max_workers: int = 4):
        """
        Initialise l'extracteur de frames YouTube
        
        Args:
            output_dir: Répertoire de sortie
            max_workers: Nombre de threads pour le traitement parallèle
        """
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Création des dossiers
        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.videos_dir, self.frames_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Répertoire de sortie: {self.output_dir}")
    def download_video(self, url: str, quality: str = "1080p", cookies: Optional[str] = None) -> Optional[Dict]:
        try:
            # Map quality to height
            quality_map = {
                "360p": 360,
                "480p": 480,
                "720p": 720,
                "1080p": 1080,
                "2k": 1440,
                "4k": 2160
            }
            height = quality_map.get(quality, 1080)
            ydl_opts = {
                'format': f'bestvideo[height<={height}]+bestaudio/best[height<={height}]',
                'outtmpl': str(self.videos_dir / '%(id)s.%(ext)s'),
                'writeinfojson': False,
                'writethumbnail': True,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
            if cookies:
                ydl_opts['cookies'] = cookies

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Extracting metadata: {url}")
                info = ydl.extract_info(url, download=False)
                logger.info(f"Available formats: {info.get('formats', [])}")

                video_id = info['id']

                if self._is_video_downloaded(video_id):
                    logger.info(f"Video {video_id} already downloaded")
                    return self._load_metadata(video_id)

                logger.info(f"Downloading: {info.get('title', 'Unknown Title')}")
                ydl.download([url])

                video_files = list(self.videos_dir.glob(f"{video_id}.*"))
                video_file = next((f for f in video_files if f.suffix in ['.mp4', '.webm', '.mkv', '.avi']), None)

                if not video_file:
                    logger.error(f"Video file not found for {video_id}")
                    return None

                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    logger.error(f"Could not open video {video_file}")
                    return None

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                metadata = VideoMetadata(
                    video_id=video_id,
                    title=info.get('title', ''),
                    description=info.get('description', '')[:1000] if info.get('description') else '',
                    duration=info.get('duration', 0),
                    upload_date=info.get('upload_date', ''),
                    uploader=info.get('uploader', ''),
                    view_count=info.get('view_count', 0),
                    like_count=info.get('like_count', 0),
                    url=url,
                    file_size=video_file.stat().st_size,
                    resolution=f"{width}x{height}",
                    fps=fps,
                    format=video_file.suffix[1:],
                    extracted_frames=0,
                    download_date=datetime.now().isoformat()
                )

                self._save_metadata(metadata)

                logger.info(f"✅ Video downloaded: {video_id} ({metadata.title})")
                return asdict(metadata)
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None

    
    def extract_frames(self, video_id: str, url: str, frame_interval: int = 30, 
                      max_frames: Optional[int] = None) -> int:
        """
        Extrait les frames d'une vidéo
        
        Args:
            video_id: ID de la vidéo
            url: URL de la vidéo
            frame_interval: Intervalle entre les frames (en nombre de frames)
            max_frames: Nombre maximum de frames à extraire
        
        Returns:
            Nombre de frames extraites
        """
        try:
            # Trouver le fichier vidéo
            video_files = list(self.videos_dir.glob(f"{video_id}.*"))
            video_file = next((f for f in video_files if f.suffix in ['.mp4', '.webm', '.mkv', '.avi']), None)
            
            if not video_file:
                logger.error(f"Fichier vidéo non trouvé pour {video_id}")
                return 0
            
            # Ouvrir la vidéo
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                logger.error(f"Impossible d'ouvrir la vidéo {video_file}")
                return 0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Extraction frames pour {video_id}: {total_frames} frames totales, FPS: {fps}")
            
            extracted_count = 0
            frame_metadata = []
            
            # Calculer les indices des frames à extraire
            frame_indices = list(range(0, total_frames, frame_interval))
            if max_frames:
                frame_indices = frame_indices[:max_frames]
            
            # Extraction avec barre de progression
            with tqdm(total=len(frame_indices), desc=f"Extraction {video_id}") as pbar:
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Nom du fichier frame
                    timestamp = frame_idx / fps
                    frame_filename = f"frame_{video_id}_{timestamp:.2f}s.jpg"
                    frame_path = self.frames_dir / frame_filename
                    
                    # Sauvegarde de la frame
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Métadonnées de la frame
                    frame_info = {
                        "frame_number": frame_idx,
                        "timestamp": timestamp,
                        "filename": frame_filename,
                        "size": frame_path.stat().st_size,
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
                    }
                    frame_metadata.append(frame_info)
                    
                    extracted_count += 1
                    pbar.update(1)
            
            cap.release()
            
            # Sauvegarde des métadonnées des frames
            frames_metadata_file = self.metadata_dir / f"{video_id}_frames.json"
            with open(frames_metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "video_id": video_id,
                    "url": url,
                    "total_frames_extracted": extracted_count,
                    "extraction_date": datetime.now().isoformat(),
                    "frame_interval": frame_interval,
                    "frames": frame_metadata
                }, f, indent=2, ensure_ascii=False)
            
            # Mettre à jour les métadonnées de la vidéo
            self._update_video_metadata(video_id, {"extracted_frames": extracted_count})
            
            logger.info(f"✅ {extracted_count} frames extraites pour {video_id}")
            return extracted_count
            
        except Exception as e:
            logger.error(f"Erreur extraction frames {video_id}: {str(e)}")
            return 0
    
    def process_urls(self, urls: List[str], quality: str = "720p", 
                    frame_interval: int = 30, max_frames: Optional[int] = None, cookies: Optional[str] = None) -> Dict:
        """
        Traite une liste d'URLs YouTube
        
        Args:
            urls: Liste des URLs YouTube
            quality: Qualité vidéo
            frame_interval: Intervalle entre frames
            max_frames: Nombre max de frames par vidéo
            cookies: Chemin vers le fichier cookies.txt pour l'authentification YouTube
        
        Returns:
            Dictionnaire avec les résultats
        """
        results = {
            "success": [],
            "errors": [],
            "total_videos": 0,
            "total_frames": 0
        }
        
        logger.info(f"Début du traitement de {len(urls)} URLs")
        
        # Téléchargement des vidéos
        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{len(urls)}] Traitement: {url}")
            
            # Téléchargement
            metadata = self.download_video(url, quality, cookies=cookies)
            if not metadata:
                results["errors"].append({"url": url, "error": "Échec téléchargement"})
                continue
            
            # Extraction des frames
            frames_count = self.extract_frames(
                metadata["video_id"], 
                url,
                frame_interval=frame_interval,
                max_frames=max_frames
            )
            
            if frames_count > 0:
                results["success"].append({
                    "url": url,
                    "video_id": metadata["video_id"],
                    "title": metadata["title"],
                    "frames_extracted": frames_count
                })
                results["total_frames"] += frames_count
            else:
                results["errors"].append({
                    "url": url, 
                    "video_id": metadata["video_id"],
                    "error": "Échec extraction frames"
                })
            
            results["total_videos"] += 1
        
        # Sauvegarde du rapport final
        report_file = self.metadata_dir / f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Traitement terminé: {len(results['success'])} succès, {len(results['errors'])} erreurs")
        logger.info(f"📊 Total: {results['total_videos']} vidéos, {results['total_frames']} frames")
        logger.info(f"📄 Rapport sauvegardé: {report_file}")
        
        return results
    
    def _is_video_downloaded(self, video_id: str) -> bool:
        """Vérifie si une vidéo est déjà téléchargée"""
        video_files = list(self.videos_dir.glob(f"{video_id}.*"))
        return any(f.suffix in ['.mp4', '.webm', '.mkv', '.avi'] for f in video_files)
    
    def _save_metadata(self, metadata: VideoMetadata):
        """Sauvegarde les métadonnées d'une vidéo"""
        metadata_file = self.metadata_dir / f"{metadata.video_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
    
    def _load_metadata(self, video_id: str) -> Optional[Dict]:
        """Charge les métadonnées d'une vidéo"""
        metadata_file = self.metadata_dir / f"{video_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _update_video_metadata(self, video_id: str, updates: Dict):
        """Met à jour les métadonnées d'une vidéo"""
        metadata = self._load_metadata(video_id)
        if metadata:
            metadata.update(updates)
            metadata_file = self.metadata_dir / f"{video_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du dataset"""
        metadata_files = list(self.metadata_dir.glob("*.json"))
        
        total_videos = 0
        total_frames = 0
        total_size = 0
        total_duration = 0
        
        for metadata_file in metadata_files:
            if metadata_file.name.endswith("_frames.json") or metadata_file.name.startswith("extraction_report"):
                continue
                
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_videos += 1
                total_frames += data.get('extracted_frames', 0)
                total_size += data.get('file_size', 0)
                total_duration += data.get('duration', 0)
        
        return {
            "total_videos": total_videos,
            "total_frames": total_frames,
            "total_size_gb": round(total_size / (1024**3), 2),
            "total_duration_hours": round(total_duration / 3600, 2),
            "avg_frames_per_video": round(total_frames / total_videos, 1) if total_videos > 0 else 0
        }

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="YouTube Frame Extractor")
    parser.add_argument("--urls", "-u", nargs="+", help="URLs YouTube à traiter")
    parser.add_argument("--file", "-f", help="Fichier contenant les URLs (une par ligne)")
    parser.add_argument("--output", "-o", default="./youtube_data", help="Répertoire de sortie")
    parser.add_argument("--quality", "-q", default="720p", choices=["360p", "480p", "720p", "1080p", "2k", "4k"], 
                       help="Qualité vidéo")
    parser.add_argument("--interval", "-i", type=int, default=30, 
                       help="Intervalle entre frames (en nombre de frames)")
    parser.add_argument("--max-frames", "-m", type=int, help="Nombre maximum de frames par vidéo")
    parser.add_argument("--stats", action="store_true", help="Afficher les statistiques")
    parser.add_argument("--cookies", help="Chemin vers le fichier cookies.txt pour l'authentification YouTube")
    
    args = parser.parse_args()
    
    # Initialisation de l'extracteur
    extractor = YouTubeFrameExtractor(output_dir=args.output)
    
    # Affichage des statistiques
    if args.stats:
        stats = extractor.get_stats()
        print("\n📊 STATISTIQUES DU DATASET")
        print("=" * 40)
        print(f"Vidéos: {stats['total_videos']}")
        print(f"Frames: {stats['total_frames']}")
        print(f"Taille: {stats['total_size_gb']} GB")
        print(f"Durée: {stats['total_duration_hours']} heures")
        print(f"Moyenne: {stats['avg_frames_per_video']} frames/vidéo")
        return
    
    # Récupération des URLs
    urls = []
    if args.urls:
        urls.extend(args.urls)
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                urls.extend(file_urls)
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {args.file}")
            return
    
    if not urls:
        logger.error("Aucune URL fournie. Utilisez --urls ou --file")
        return
    
    # Traitement
    logger.info(f"Démarrage avec {len(urls)} URLs")
    results = extractor.process_urls(
        urls=urls,
        quality=args.quality,
        frame_interval=args.interval,
        max_frames=args.max_frames,
        cookies=args.cookies
    )
    
    # Résumé final
    print("\n🎯 RÉSUMÉ FINAL")
    print("=" * 40)
    print(f"✅ Succès: {len(results['success'])}")
    print(f"❌ Erreurs: {len(results['errors'])}")
    print(f"📹 Total vidéos: {results['total_videos']}")
    print(f"🖼️  Total frames: {results['total_frames']}")
    
    if results['errors']:
        print("\n❌ ERREURS:")
        for error in results['errors']:
            print(f"  - {error['url']}: {error['error']}")

if __name__ == "__main__":
    main()